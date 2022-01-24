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
"""

FLAGS = flags.FLAGS
APP_ID = "application_1641760327031_0006"
#flags.DEFINE_string("a", "dqn", "agent name")
#flags.DEFINE_integer("b", 1, "threshold for huber loss")
#flags.DEFINE_float("c", 0.99, "gamma discount")


def get_all_stages() -> None:
    #resp = requests.get(f"http://localhost:18080/api/v1/applications/{APP_ID}/allexecutors")
    resp = requests.get(
        f"http://localhost:18080/api/v1/applications/{APP_ID}/stages")

    #print(resp.json())
    js = resp.json()

    print(len(js))

    for i in range(len(js)):
        print("==============")
        print(js[i])
        print("==============")


def get_a_stage() -> None:
    #resp = requests.get(f"http://localhost:18080/api/v1/applications/{APP_ID}/allexecutors")
    resp = requests.get(
        f"http://localhost:18080/api/v1/applications/{APP_ID}/stages/0/0")

    #print(resp.json())
    js = resp.json()
    #print(js)
    print([key for key, val in js.items()])


def get_per_task() -> None:
    #resp = requests.get(f"http://localhost:18080/api/v1/applications/{APP_ID}/allexecutors")
    resp = requests.get(
        f"http://localhost:18080/api/v1/applications/{APP_ID}/stages/0/0/taskSummary"
    )

    #print(resp.json())
    js = resp.json()
    print(js)


def executors_metric() -> None:
    #resp = requests.get(f"http://localhost:18080/api/v1/applications/{APP_ID}/allexecutors")
    resp = requests.get(
        f"http://localhost:18080/api/v1/applications/{APP_ID}/executors")

    js = resp.json()
    for i in range(len(js)):
        print("==============")
        print(js[i])
        print("==============")


def main(_):
    get_all_stages()
    get_a_stage()
    get_per_task()
    #executors_metric()


if __name__ == "__main__":
    app.run(main)
