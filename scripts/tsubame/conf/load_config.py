import csv, pathlib
_CSV = pathlib.Path(__file__).parent / "tasks_runtime.csv"
tasks = {}
with _CSV.open(newline="") as f:
        for row in csv.DictReader(filter(lambda r: not r.startswith("#"), f),
                                  delimiter=",", skipinitialspace=True):
            row = {k: v.strip() for k, v in row.items()}
            tasks[row["key"]] = row

def task_script(k):    return tasks[k]["script"]
def task_result(k):    return tasks[k]["result_dir"]
def hrt(node, k):      return tasks[k]["hrt_q" if node=="node_q" else "hrt_f"]
def all_tasks():       return list(tasks)
