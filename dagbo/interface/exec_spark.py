import os
import subprocess
from time import sleep
from copy import deepcopy
from typing import Union
from .metrics_extractor import extract_app_id, extract_throughput

# example:
#rc = subprocess.call("benchmarks/sleep.sh", shell=True)
#print(rc)
#
#rc = subprocess.run(["ls", "-l"])
#print(rc)
#print("...")
#
#rc = subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
#print(rc)
"""
hardcode param conversion rules
    using only a subset of those param is allowed
    but new param must be hardcode

spawn a child process and execute spark job with given parameters
"""

# this will be written to spark.conf regardless input parameters
CONST_WRITE = {
    "hibench.spark.home": "/local/scratch/opt/spark-2.4.5-bin-hadoop2.7",
    "hibench.spark.master": "yarn",
    "spark.eventLog.enabled": "true",
    "spark.local.dir":
    "/local/scratch/spark_tmp_dir",  # to store intermediate data

    # for continuous perf model
    "spark.speculation": "true",
    "spark.shuffle.compress": "false",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.rdd.compress": "true",
}

# map from param to actual spark config name
NAME_MAPPING = {
    "executor.num[*]": "hibench.yarn.executor.num",
    "executor.cores": "hibench.yarn.executor.cores",
    "shuffle.compress": "spark.shuffle.compress",
    "executor.memory": "spark.executor.memory",
    "memory.fraction": "spark.memory.fraction",
    "spark.serializer": "spark.serializer",
    "rdd.compress": "spark.rdd.compress",
    "default.parallelism": "spark.default.parallelism",
    "shuffle.spill.compress": "spark.shuffle.spill.compress",
    "spark.speculation": "spark.speculation",
}

# NOTE: possibly the most important mapping, scale param range [0, 1] back to their original values
SCALE_MAPPING = {
    "executor.num[*]": 4,
    "executor.cores": 4,
    "executor.memory": 4,
    "default.parallelism": 16,
}

# max(possible lowest value, actual val)
MIN_MAPPING = {
    "executor.num[*]": 2,
    "executor.cores": 1,
    "executor.memory": 1,
    "default.parallelism": 2,
    "memory.fraction": 0.1,
}

MAX_MAPPING = {
    "memory.fraction": 0.95,
}

# 0.5 -> 0, 0.51 -> 1
ROUND_MAPPING = {
    "executor.num[*]": "int",
    "executor.cores": "int",
    "shuffle.compress": "bool",
    "executor.memory": "int",
    "memory.fraction": "float",
    "spark.serializer": "bool",
    "rdd.compress": "bool",
    "default.parallelism": "int",
    "shuffle.spill.compress": "bool",
    "spark.speculation": "bool",
}

# spec that needs unit mapping, e.g. '4' -> '4g'
UNIT_MAPPING = {
    "spark.executor.memory": 0,
}

# 0 -> false, 1 -> true
BOOL_MAPPING = {
    "spark.shuffle.compress": 0,
    "spark.rdd.compress": 0,
    "spark.shuffle.spill.compress": 0,
    "spark.speculation": 0,
}

#
CAT_MAPPING = {
    "spark.serializer": [
        "org.apache.spark.serializer.KryoSerializer",
        "org.apache.spark.serializer.JavaSerializer"
    ]
}

# they should mean the same thing
DUPLICATE_MAPPING = {
    "hibench.yarn.executor.num": "spark.executor.instances",
}


def call_spark(param: dict[str, Union[float, int]], file_path: str,
               exec_path: str) -> None:
    """
    call hibench spark benchmarks with the given parameters
    Args:
        param: param file generated by bo
        file_path: the abs path to put the configuration file
        exec_path: the abs path to executable
    """

    # pre-processing param
    param_ = _pre_process(deepcopy(param))

    # write spec according to param
    _write_spec_from_param(param_, file_path)

    # exec
    _exec(exec_path)

    return None


def _exec(exec_path: str) -> None:
    """
    must given enough time to properly write log & execute
    """
    rc = subprocess.run([exec_path])
    if rc.returncode != 0:
        print("stderr: ")
        print(rc.stderr)
        raise RuntimeError("exec spark return non-zero code")

    # give time to write file to history server
    sleep(15)
    return None


def _write_spec_from_param(param: dict[str, str], file_path: str) -> None:
    # remove if exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # write config
    with open(file_path, "a") as f:
        for key, val in CONST_WRITE.items():
            f.write(key)
            f.write("\t")
            f.write(val)
            f.write("\n")

        for key, val in param.items():
            f.write(key)
            f.write("\t")
            f.write(val)
            f.write("\n")

    return None


def _pre_process(param: dict[str, float]) -> dict[str, str]:
    """
    name mapping & unit mapping & bool mapping &
    """

    # scale mapping
    for key, val in param.items():
        if key in SCALE_MAPPING:
            param[key] = val * SCALE_MAPPING[key]

    # min mapping
    for key, val in param.items():
        if key in MIN_MAPPING:
            param[key] = max(val, MIN_MAPPING[key])

    # max mapping
    for key, val in param.items():
        if key in MAX_MAPPING:
            param[key] = min(val, MAX_MAPPING[key])

    # round mapping
    for key, val in param.items():
        if ROUND_MAPPING[key] == "float":
            continue
        elif ROUND_MAPPING[key] == "int":  # don't use int(float val)
            param[key] = round(val)
        elif ROUND_MAPPING[key] == "bool":
            if 0 <= val <= 1:
                param[key] = round(val)
            else:
                raise ValueError(
                    f"try to ROUND bool param {key} with value {val}")
        else:
            raise TypeError(f"unknown param {key} with value {val}")

    # name mapping
    param_ = {}
    for key, val in param.items():
        param_[NAME_MAPPING[key]] = str(val)

    # duplicate mapping
    add = {}
    for key, val in param_.items():
        if key in DUPLICATE_MAPPING:
            add[DUPLICATE_MAPPING[key]] = val
    for k, v in add.items():
        param_[k] = v

    # cat mapping
    for key in list(param_.keys()):
        if key in CAT_MAPPING:
            if param_[key] == "0":
                param_[key] = CAT_MAPPING[key][0]
            elif param_[key] == "1":
                param_[key] = CAT_MAPPING[key][1]
            else:
                raise ValueError(f"unsupported param {key} val {param_[key]}")

    # perform unit mapping, e.g. '4' -> '4g'
    for key in list(param_.keys()):
        if key in UNIT_MAPPING:
            param_[key] = str(param_[key]) + "g"

    # perform bool mapping, e.g. '0' -> false
    for key in list(param_.keys()):
        if key in BOOL_MAPPING:
            val = param_[key]
            if val == "0":
                param_[key] = "false"
            elif val == "1":
                param_[key] = "true"
            else:
                raise ValueError("unknown boolean val")

    return param_
