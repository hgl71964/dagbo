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


def request_history_server(base_url,
                           app_id) -> dict[str, dict[str, Union[int, float]]]:
    """
    send request to spark's history_server, and extract monitoring metrics

    Returns:
        exec_map. key: executor id - val: k, v pair of metric (map[str, union[int, float]])
    """
    stage_ids = _get_stages(base_url, app_id)
    exec_map = _get_executors_metric(base_url, app_id, stage_ids)
    return _post_processing(exec_map)


def _post_processing(exec_map):
    """
    unit conversion etc...
    """
    #print(exec_map)
    #  TODO
    return exec_map


def _get_executors_metric(base_url, app_id, stage_ids):
    exec_metric_list = []
    for s_id in stage_ids:
        resp = requests.get(
            f"{base_url}/api/v1/applications/{app_id}/stages/{s_id}")

        js = resp.json()

        if len(js) > 1:  # more than one attempt, raise
            raise RuntimeError(f"stage {s_id} has more than one attempt")

        js = js[0]
        stage_exec_map = _parse_per_stage_json(js)
        exec_metric_list.append(stage_exec_map)

    return _aggregate_across_stages(exec_metric_list)


def _aggregate_across_stages(exec_metric_list):
    """
    impl aggregation logic across stages
    Args:
        exec_metric_list: a list of exec_map, where each exec_map records per-stage executor metrics
    Returns:
        aggregated exec_map. key: executor id - val: k, v pair of metric (map[str, union[int, float]])
    """
    n = len(exec_metric_list)
    if n < 1:
        raise RuntimeError("num stage < 1?")

    # init exec_map & its id
    exec_map = {}
    for stage in exec_metric_list:
        for idx in stage:
            if idx not in exec_map:
                exec_map[idx] = {
                    "taskTime": 0,
                    "memoryBytesSpilled": 0,
                    "diskBytesSpilled": 0,
                    "executorDeserializeTime": 0,
                    "executorDeserializeCpuTime": 0,
                    "executorRunTime": 0,
                    "executorCpuTime": 0,
                    "jvmGcTime": 0,
                }

    # aggregate
    for stage in exec_metric_list:
        for idx in stage.keys():
            exec_map[idx]["taskTime"] += stage[idx]["taskTime"]
            exec_map[idx]["memoryBytesSpilled"] += stage[idx][
                "memoryBytesSpilled"]
            exec_map[idx]["diskBytesSpilled"] += stage[idx]["diskBytesSpilled"]

            # list, where each element is task-level metric
            exec_map[idx]["executorDeserializeTime"] += sum(
                stage[idx]["executorDeserializeTime"])
            exec_map[idx]["executorDeserializeCpuTime"] += sum(
                stage[idx]["executorDeserializeCpuTime"])
            exec_map[idx]["executorRunTime"] += sum(
                stage[idx]["executorRunTime"])
            exec_map[idx]["executorCpuTime"] += sum(
                stage[idx]["executorCpuTime"])
            exec_map[idx]["jvmGcTime"] += sum(stage[idx]["jvmGcTime"])
    return exec_map


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
            raise RuntimeWarning(
                f"task {task['taskId']} has status {task['status']} and speculative {task['speculative']}"
            )

        exec_map = _add_metric(exec_map, task)

    return exec_map


def _add_metric(exec_map, task):
    idx = task["executorId"]
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
            print(f"skipped stage {stageid} with status {status}")
            continue
        else:
            stage_ids.append(stageid)

    #print(stage_ids)
    return stage_ids


if __name__ == "__main__":
    app.run(main)
