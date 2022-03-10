import re
from warnings import warn
import numpy as np

import requests
from absl import app
from absl import flags
from tqdm import tqdm
from typing import Union
"""
workflow to interface with hiBench & Spark

1. at ${hiBench home}/report/wordcount/spark/bench.log
    we can retrieve the submitted application ID

2. to see spark metrics,
    0. set up `spark.eventLog.enabled  true` at ${hiBench home}/conf/spark.conf
    1. start-history-server.sh
    2. executors metrics available at:
        JSON format at http://localhost:18080/app/v1/applications/{XXX _APPID}/executors

    3. per-stages info:
        .../applications/{XXX _APPID}/stages

    3. per-task metric available at:
        .../applications/{XXX _APPID}/stages/[stage-id]

NOTE: the description for each metric: https://spark.apache.org/docs/2.4.5/monitoring.html

Example url:
     curl http://localhost:18080/api/v1/applications/application_1641844906451_0006/stages/
"""

FLAGS = flags.FLAGS


def main(_):
    app_id = "application_1641844906451_0006"
    base_url = "http://localhost:18080"
    request_history_server(base_url, app_id)


def extract_and_aggregate(params: dict[str, float],
                          train_inputs_dict: dict[str, np.ndarray],
                          train_targets_dict: dict[str, np.ndarray],
                          base_url: str) -> float:
    """
    extract & aggregation metric & populate data
    """
    # throughput
    #val = extract_throughput(hibench_report_path)
    #val = float(val)
    #app_id = extract_app_id(base_url)

    app_id, val = extract_duration_app_id(base_url)
    val = float(val)

    metric_list = request_history_server(base_url, app_id)

    # agg
    agg_m = _aggregation(metric_list)
    ## add final obj
    agg_m["duration"] = val
    ## unit conversion
    agg_m = _post_processing(agg_m)

    # populate input
    for k, v in params.items():
        if k in train_inputs_dict:
            train_inputs_dict[k] = np.append(train_inputs_dict[k], v)
        else:
            train_inputs_dict[k] = v
    # populate output
    for k, v in agg_m.items():
        if k in train_targets_dict:
            train_targets_dict[k] = np.append(train_targets_dict[k], v)
        else:
            train_targets_dict[k] = v
    return val


def _aggregation(exec_metric_list: list[dict[str, list[float]]]) -> dict:
    straggler_metric_list = []
    num_stages = len(exec_metric_list)
    if num_stages < 1:
        raise RuntimeError("num stage < 1?")

    # aggregate across executors, i.e. for each stage, find the straggler (max taskTime)
    for stage_dict in exec_metric_list:
        straggler_id = None
        straggler_time = -float("inf")
        for executor_id, metric in stage_dict.items():
            t = metric["taskTime"]
            if t > straggler_time:
                straggler_id = executor_id
                straggler_time = t
        if straggler_id is None:
            raise ValueError("unable to find straggler")

        straggler_metric_list.append(stage_dict[straggler_id])

    # aggregate across stages, sum
    d = {}
    for per_stage_dict in straggler_metric_list:
        for metric_name, val in per_stage_dict.items():
            if isinstance(val, list):
                val = sum(val)
            if metric_name not in d:
                d[metric_name] = val
            else:
                d[metric_name] += val
    return d


def _post_processing(metric_map: dict[str, float]) -> dict[str, float]:
    """
    unit conversion etc...

    time:
    taskTime, executorDeserializeTime, executorRunTime, jvmGcTime are milliseconds
    executorDeserializeCpuTime, executorCpuTime are nanoseconds
    """
    metric_map["executorDeserializeCpuTime"] = metric_map[
        "executorDeserializeCpuTime"] * 1e-6
    metric_map["executorCpuTime"] = metric_map["executorCpuTime"] * 1e-6
    return metric_map


"""
App-level extraction
"""


def extract_duration_app_id(base_url: str) -> tuple[str, str]:
    resp = requests.get(f"{base_url}/api/v1/applications/")

    if resp.status_code != 200:
        raise RuntimeError(
            f"get request not ok - status code: {resp.status_code} - {base_url}/api/v1/applications/{app_id}/stages"
        )
    js = resp.json()

    # XXX consider the first item the latest?
    latest = js[0]
    app_id = latest["id"]
    print(f"get app id: {app_id}")

    n = len(latest["attempts"])
    if n > 1:
        warn(f"{app_id} has more than one attempt")

    duration = None
    for i in range(n):
        tmp = latest["attempts"][i]
        if tmp["completed"] == "true":
            duration = tmp["duration"]

    if duration is None:
        raise RuntimeError(
            f"unable to get duration from {base_url}/api/v1/applications/{app_id}"
        )
    return app_id, duration


def extract_throughput(hibench_report_path: str) -> str:
    """
    according to hibench report log text structure
        by default the last line is the latest run results
    """
    with open(hibench_report_path, "r") as f:
        lines = f.readlines()
        l = lines[-1].strip().split()

        #for line in lines:
        #    print(line.strip().split())
    return l[-2]


def extract_app_id(log_path: str) -> str:
    app_id = None
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split()
            if "impl.YarnClientImpl:" not in l:
                continue
            app_id = l[-1]
            break

    if not app_id:
        raise ValueError("unable to find application id")
    return app_id


def request_history_server(base_url, app_id) -> list[dict]:
    """
    send request to spark's history_server, and extract monitoring metrics

    Returns:
        exec_metric_list. list of per-stage exec_map, where k: executor id - v: list of metric values
    """
    stage_ids = _get_stages(base_url, app_id)
    exec_metric_list = _get_executors_metric(base_url, app_id, stage_ids)
    return exec_metric_list


"""
fine-grained metric extraction
"""


def _get_executors_metric(base_url, app_id,
                          stage_ids) -> list[dict[str, list[float]]]:
    exec_metric_list = []
    for s_id in stage_ids:
        resp = requests.get(
            f"{base_url}/api/v1/applications/{app_id}/stages/{s_id}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"get request not ok - status code: {resp.status_code} - {base_url}/api/v1/applications/{app_id}/stages"
            )

        js = resp.json()

        if len(js) > 1:  # more than one attempt, raise
            raise RuntimeError(
                f"stage {s_id} has more than one attempt, could be a worker lost connection"
            )

        js = js[0]
        stage_exec_map = _parse_per_stage_json(js)
        exec_metric_list.append(stage_exec_map)

    return exec_metric_list


def _parse_per_stage_json(js) -> dict:
    #for i in js:
    #    print(i)
    # exec_map already contains stage-level:
    # `taskTime`, `shuffle.*`, `input/output.*`, `memoryBytesSpilled`, `diskBytesSpilled`
    exec_map = js["executorSummary"]

    all_tasks = js["tasks"]
    tasks_list = list(all_tasks.keys())
    exec_list = list(exec_map.keys())

    for task_id in tasks_list:
        task = all_tasks[task_id]
        if task["attempt"] > 0 or task["status"] != "SUCCESS":
            tid = task['taskId']
            ts = task['status']
            tsp = task['speculative']
            warn(f"task {tid} has status {ts} and speculative {tsp}")

        exec_map = _append_metric(exec_map, task)

    return exec_map


def _append_metric(exec_map, task):
    """
    add per-task level metric, to executor summary dict
    """
    idx = task["executorId"]

    # XXX cases where the task's metrics cannot be extracted, just ignore for now
    if "taskMetrics" not in task:
        task_id = task["taskId"]
        task_host = task["host"]
        task_status = task["status"]
        task_speculative = task["speculative"]
        warn(
            f"dead task - id: {task_id} - host: {task_host} - status: {task_status} - speculative: {task_speculative}"
        )
        return exec_map

    task_metric = task["taskMetrics"]
    if "executorDeserializeTime" not in exec_map[idx]:
        exec_map[idx]["executorDeserializeTime"] = [
            task_metric["executorDeserializeTime"]
        ]
    else:
        exec_map[idx]["executorDeserializeTime"].append(
            task_metric["executorDeserializeTime"])

    if "executorDeserializeCpuTime" not in exec_map[idx]:
        exec_map[idx]["executorDeserializeCpuTime"] = [
            task_metric["executorDeserializeCpuTime"]
        ]
    else:
        exec_map[idx]["executorDeserializeCpuTime"].append(
            task_metric["executorDeserializeCpuTime"])

    if "executorRunTime" not in exec_map[idx]:
        exec_map[idx]["executorRunTime"] = [task_metric["executorRunTime"]]
    else:
        exec_map[idx]["executorRunTime"].append(task_metric["executorRunTime"])

    if "executorCpuTime" not in exec_map[idx]:
        exec_map[idx]["executorCpuTime"] = [task_metric["executorCpuTime"]]
    else:
        exec_map[idx]["executorCpuTime"].append(task_metric["executorCpuTime"])

    if "jvmGcTime" not in exec_map[idx]:
        exec_map[idx]["jvmGcTime"] = [task_metric["jvmGcTime"]]
    else:
        exec_map[idx]["jvmGcTime"].append(task_metric["jvmGcTime"])
    return exec_map


def _get_stages(base_url, app_id):
    """
    get all stages id. if a stage is not completed, it is skipped
    """
    resp = requests.get(f"{base_url}/api/v1/applications/{app_id}/stages")

    if resp.status_code != 200:
        raise RuntimeError(
            f"get request not ok - status code: {resp.status_code} - {base_url}/api/v1/applications/{app_id}/stages"
        )

    js = resp.json()

    #for i in range(len(js)):
    #    print("==============")
    #    print(js[i])
    #    print("==============")

    stage_ids = []
    for i in range(len(js)):
        record = js[i]
        status = record["status"]
        stageid = record['stageId']
        if status != "COMPLETE":
            warn(f"skipped stage {stageid} with status {status}")
            continue
        else:
            stage_ids.append(stageid)

    #print(stage_ids)
    return stage_ids


if __name__ == "__main__":
    app.run(main)
