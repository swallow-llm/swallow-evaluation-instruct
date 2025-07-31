import subprocess
import re
import json
import os

#======================================================================
# qstat.py
# How to use: 
# 
# python scripts/qsub/utils/qstat.py


#======================================================================


def get_qstat_output():
    try:
        result = subprocess.run(['qstat', '-f'], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running qstat:", e)
        return ""

def parse_qstat_output(qstat_output):
    jobs = re.split(r'\n\s*Job Id:\s*', qstat_output.strip())
    job_infos = []

    for job_raw in jobs:
        if not job_raw.strip():
            continue

        lines = job_raw.splitlines()
        first_line = lines[0]
        job_id = first_line.strip()
        
        job_id = re.search(r'(\d+)', job_id)
        if job_id: 
            job_id = job_id.group(1)
        else:
            continue
        job_text = '\n'.join(lines[1:])

        # 実行中ジョブのみ抽出
        state_match = re.search(r'job_state\s*=\s*R', job_text)
        if not state_match:
            continue

        job_name = re.search(r'Job_Name\s*=\s*(.+)', job_text)
        used_time = re.search(r'resources_used.walltime\s*=\s*([0-9:]+)', job_text)
        max_time = re.search(r'Resource_List.walltime\s*=\s*([0-9:]+)', job_text)
        job_state = re.search(r'job_state\s*=\s*(\w+)', job_text)
        # モデル名の前後の空白・改行・インデントを取り除いて連結
        flattened = ' '.join(line.strip() for line in job_text.splitlines())
        flattened = re.sub(r'\s+', ' ', flattened)  # 空白正規化

        model = re.search(r'--model-name\s+(.+?)(?=\s+--|-\s+-)', flattened, re.DOTALL)
        job_infos.append({
            "Job ID": job_id,
            "Job Name": job_name.group(1).strip() if job_name else '',
            "Model Name": model.group(1).strip().replace(' ','') if model else 'N/A',
            "Elapsed Time": used_time.group(1) if used_time else 'N/A',
            "Max Time": max_time.group(1) if max_time else 'N/A',
            "Job State": job_state.group(1) if job_state else 'N/A'
        })

    return job_infos

def main():

    qstat_output = get_qstat_output()
    jobs = parse_qstat_output(qstat_output)

    if not jobs:
        print("実行中のジョブが見つかりません。")
        return

    # jobsをモデル名でソート
    jobs.sort(key=lambda x: x['Model Name'])
    current_model = jobs[0]['Model Name']

    print(f"{'Job ID':<10} {'State':<5} {'Elapsed':<10} {'Max Time':<10} {'Job Name':<30} {'Model Name':<30}")
    print("-" * 110)
    for job in jobs:
        if job['Model Name'] != current_model:
            current_model = job['Model Name']
            print("-" * 110)
        print(f"{job['Job ID']:<10} {job['Job State']:<5} {job['Elapsed Time']:<10} {job['Max Time']:<10} {job['Job Name']:<30} {job['Model Name']:<30}")
    print(f"🐦 ..... Total Jobs: {len(jobs)}")

    # ヒストリーをjsonで保存（スクリプトと同じディレクトリ配下に保存）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    history_path = os.path.join(script_dir, 'qstat_history.json')

    prev_jobs = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
                prev_jobs = prev.get('jobs', [])
        except Exception as e:
            print(f"[WARN] qstat_history.jsonの読み込みに失敗: {e}")

    # Job IDで比較
    prev_job_dict = {j['Job ID']: j for j in prev_jobs}
    curr_job_dict = {j['Job ID']: j for j in jobs}

    gone_jobs = [j for jid, j in prev_job_dict.items() if jid not in curr_job_dict]
    changed_jobs = []
    for jid in curr_job_dict:
        if jid in prev_job_dict and curr_job_dict[jid] != prev_job_dict[jid]:
            changed_jobs.append((prev_job_dict[jid], curr_job_dict[jid]))

    if prev_jobs:
        if gone_jobs:
            print("\n=== 終了・消滅ジョブ ===")
            

            # qstat -xfで詳細取得
            job_ids = [j['Job ID'] for j in gone_jobs]
            if job_ids:
                import subprocess
                try:
                    result = subprocess.run(['qstat', '-xf'] + job_ids, capture_output=True, text=True, check=True)
                    output = result.stdout
                    # 各Job Idごとにパース
                    for job_id in job_ids:
                        # PBSのJob Idは数字だけでなくサフィックスがつく場合があるので部分一致で探す
                        m = re.search(rf'Job Id:.*{job_id}.*?\n(.*?)(?=\nJob Id:|\Z)', output, re.DOTALL)
                        if not m:
                            print(f"  [WARN] qstat -xfで{job_id}の詳細が見つかりません")
                            continue
                        job_block = m.group(0)
                        walltime = re.search(r'resources_used.walltime\s*=\s*([0-9:]+)', job_block)
                        job_state = re.search(r'job_state\s*=\s*(\w+)', job_block)
                        exit_status = re.search(r'Exit_status\s*=\s*(-?\d+)', job_block)
                        
                        walltime = walltime.group(1) if walltime else 'N/A'
                        job_state = job_state.group(1) if job_state else 'N/A'
                        exit_status = exit_status.group(1) if exit_status else 'N/A'
                        
                        i = next((i for i, j in enumerate(gone_jobs) if j['Job ID'] == job_id), None)
                        if exit_status == '0':
                            print(f"👍Job ID: {job_id}, Walltime: {walltime}, model: {gone_jobs[i]['Model Name']}, task: {gone_jobs[i]['Job Name']}")
                        else:
                            print(f"❌Job ID: {job_id}, Walltime: {walltime}, model: {gone_jobs[i]['Model Name']}, task: {gone_jobs[i]['Job Name']}, Exit Status: {exit_status}")
                except Exception as e:
                    print(f"  [WARN] qstat -xf実行失敗: {e}")

    # 保存
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'jobs': jobs
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
