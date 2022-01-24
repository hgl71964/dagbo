import requests
from absl import app
from absl import flags
from tqdm import tqdm
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
        .../applications/{XXX _APPID}/stages/[stage-id]/[stage-attempt-id]/taskSummary

NOTE: the description for each metric: https://spark.apache.org/docs/2.4.5/monitoring.html

Example url:
     curl http://localhost:18080/api/v1/applications/application_1641844906451_0006/stages/
"""

FLAGS = flags.FLAGS


def main(_):
    app_id = "application_1641844906451_0006"
    request_history_server(app_id)


def request_history_server(app_id):
    stage_ids = _get_stages(app_id)

    _get_executors_metric(app_id, stage_ids)

    return


def _get_executors_metric(app_id, stage_ids):
    for s_id in stage_ids:
        resp = requests.get(
            f"http://localhost:18080/api/v1/applications/{app_id}/stages/{s_id}"
        )
        js = resp.json()

        if len(js) > 1:  # more than one attempt, raise
            raise RuntimeError(f"stage {s_id} has more than one attempt")

        js = js[0]
        _parse_stage_js(js)

        break
    return


def _parse_stage_js(js):
    for i in js:
        print(i)

    all_tasks = js["tasks"]
    tasks_list = list(js["tasks"].keys())
    all_exec = js["executorSummary"]
    exec_list = list(js["executorSummary"].keys())

    print(tasks_list)
    print(exec_list)
    print(js["executorSummary"])

    # TODO
    return


def _get_stages(app_id):
    """
    get all stages id. if a stage is not completed, it is skipped
    """
    resp = requests.get(
        f"http://localhost:18080/api/v1/applications/{app_id}/stages")
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
