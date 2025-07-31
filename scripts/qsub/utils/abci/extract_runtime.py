import os
import re
import subprocess
import argparse
# HOW TO USE:
# runtime.shの方に書いてあります。


# ========== Script to extract runtime from qsub jobs =========

# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument('--repo_path', default="/home")
parser.add_argument('--hosting', default="hosted_vllm")
parser.add_argument('--model_id', default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
parser.add_argument('--setting_name', default="reasoning_swallow")
parser.add_argument('--return_seconds', action='store_true', help="Return runtime in seconds instead of minutes")
args = parser.parse_args()

REPO_PATH = args.repo_path
HOSTING = args.hosting
MODEL_ID = args.model_id
SETTING_NAME = args.setting_name

base_dir = os.path.join(REPO_PATH, "results", HOSTING, MODEL_ID, SETTING_NAME)

task_runtime = {k: -1 for k in [
    "en/aime_2024_2025",
    "en/hellaswag",
    "en/math_500",
    "en/mmlu_pro",
    "en/mt_bench",
    "en/gpqa_diamond",
    "en/livecodebench_v5_v6",
    "en/mmlu",
    "en/mmlu_prox",
    "ja/gpqa",
    "ja/jemhopqa_cot",
    "ja/mifeval",
    "ja/mmlu_prox",
    "ja/wmt20_en_ja",
    "ja/humaneval",
    "ja/math_100",
    "ja/mmlu",
    "ja/mt_bench",
    "ja/wmt20_ja_en",
]}

# 前提: タスク名リストとベースディレクトリがある
task_names = list(task_runtime.keys())

pbs1_ou_pattern = re.compile(r"(\d+)\.pbs1.OU$")

def parse_walltime(walltime_str):
    if args.return_seconds:
        h, m, s = map(int, walltime_str.strip().split(":"))
        return h * 3600 + m * 60 + s
    else:
        return walltime_str.strip()  



# まず全タスクの最新job_idを集める
task_to_jobid = {}
for task in task_names:
    search_dir = os.path.join(base_dir, task)
    if not os.path.exists(search_dir):
        continue
    job_files = [f for f in os.listdir(search_dir) if pbs1_ou_pattern.match(f)]
    if not job_files:
        continue
    job_files.sort()
    job_file = job_files[-1]
    m = pbs1_ou_pattern.match(job_file)
    if m:
        job_id = m.group(1)
        task_to_jobid[task] = job_id

print(f"Found {len(task_to_jobid)} tasks with job IDs.")

# 一括でqstat -xf
all_jobids = list(task_to_jobid.values())
jobid_to_task = {v: k for k, v in task_to_jobid.items()}
queue_name = ''
queue_found = False
if all_jobids:
    try:
        qstat_out = subprocess.check_output(["qstat", "-xf"] + all_jobids, text=True)
        # 各job_idごとにブロックを分割
        job_blocks = re.split(r"\n(?=Job Id: )", qstat_out)
        for block in job_blocks:
            m = re.match(r"Job Id: (\d+)", block)
            if not m:
                continue
            job_id = m.group(1)
            task = jobid_to_task.get(job_id)
            if not task:
                continue
            exit_status_match = re.search(r'Exit_status\s*=\s*([0-9]+)', block)
            if exit_status_match and exit_status_match.group(1) == '0':
                walltime = -1
                walltime_match = re.search(r'<resources_used.walltime>([0-9:]+)</resources_used.walltime>', block)
                if walltime_match:
                    walltime = parse_walltime(walltime_match.group(1))
                else:
                    walltime_match2 = re.search(r'resources_used.walltime\s*=\s*([0-9:]+)', block)
                    if walltime_match2:
                        walltime = parse_walltime(walltime_match2.group(1))
                task_runtime[task] = walltime
                # queue名を最初の1つだけ取得
                if not queue_found:
                    queue_match = re.search(r'queue\s*=\s*([\w-]+)', block)
                    if queue_match:
                        queue_name = queue_match.group(1)
                        queue_found = True
            else:
                task_runtime[task] = -1
    except subprocess.CalledProcessError:
        for task in task_to_jobid:
            task_runtime[task] = -1

# 結果出力
print("\n【タスク別 実行時間（秒）】")
for task, t in task_runtime.items():
    if t == -1:
        print(f"{task:30s} - No valid job (or not finished successfully).")
    else:
        print(f"{task:30s} - {t} 秒")
        
        
# 結果出力
print("="*10)
print("【CSVコピー用】")
task_items = sorted(task_runtime.items(), key=lambda x: x[0])
header = f"model_id,queue,setting_name,hosting," + ",".join([f"{task}" for task, _ in task_items])
row = f"{MODEL_ID},{queue_name},{SETTING_NAME},{HOSTING},"
for task, _ in task_items:
    row += f"{task_runtime[task]},"
print(header.rstrip(','))
print(row.rstrip(','))