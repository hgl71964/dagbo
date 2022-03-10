import datetime
import pandas as pd

from hyperopt import hp, tpe, rand
from hyperopt.base import Trials, trials_from_docs, JOB_STATE_DONE
import ax
from ax import Experiment, RangeParameter


def get_model(tuner: str):
    if tuner == "rand":
        return rand.suggest
    elif tuner == "tpe":
        return tpe.suggest
    else:
        raise ValueError("unable to recognize tuner")


def search_space_from_ax_experiment(exp: Experiment) -> dict:
    """
    convert ax-platform's experiment search space
        -> hyperopt's search space

    Example:
        search_space = {
            'x': hp.uniform('x', -6, 6),
            'y': hp.uniform('y', -6, 6),
        }
    """
    hyperopt_seach_space = {}
    for k, v in exp.parameters.items():
        assert isinstance(
            v, RangeParameter), "only support RangeParameter for now"
        assert k == v.name, "parameter name and RangeParameter does not match"
        assert v.parameter_type == ax.ParameterType.FLOAT, "only support float now, maybe remove in the future"

        hyperopt_seach_space[k] = hp.uniform(k, v.lower, v.upper)
        #print(k, v.lower, v.upper)
    return hyperopt_seach_space


def build_trials_from_sobol(exp: Experiment, minimize: bool) -> Trials:
    """

    Example format (docs):
    results = [
        {'loss': 10., 'status': 'ok'}, ...
    ]
    miscs = [
                {
                    "tid":0,
                    'cmd': ('domain_attachment', 'FMinIter_Domain'),
                    'idxs': {'x': [0], 'y': [0]},
                    'vals': {'x': [-1.], 'y': [-1.]},
                },
                {
                    "tid":1,
                    'cmd': ('domain_attachment', 'FMinIter_Domain'),
                    'idxs': {'x': [1], 'y': [1]},
                    'vals': {'x': [-2.], 'y': [-2.]},
                }, ...
            ]
    """

    # retrieve sobol
    df = exp.fetch_data().df.set_index("arm_name")
    arms_df = pd.DataFrame.from_dict(
        {k: v.parameters
         for k, v in exp.arms_by_name.items()}, orient="index")

    join_df = df.join(arms_df)  # will join according to arm_idx
    n = len(join_df)
    #print(join_df)
    #print(len(join_df))

    # retrieve param names
    params_name = list(exp.parameters.keys())

    # build docs
    tids = [i for i in range(n)]
    specs = [None for i in range(n)]

    # NOTE: the reward needs to flip sign, as hyperopt by default perform minimization
    if minimize:
        results = [{
            "loss": i,
            "status": "ok"
        } for i in join_df["mean"].to_list()]
    else:
        results = [{
            "loss": -i,
            "status": "ok"
        } for i in join_df["mean"].to_list()]
    miscs = [{
        "tid": i,
        "cmd": ("domain_attachment", "FMinIter_Domain"),
        "idxs": {param: [i]
                 for param in params_name},
        "vals": {param: [join_df[param][i]]
                 for param in params_name},
    } for i in range(n)]

    docs = []
    for tid, spec, result, misc in zip(tids, specs, results, miscs):
        doc = {
            "state": JOB_STATE_DONE,  # so hyperopt will not do this again
            "tid": tid,
            "spec": spec,
            "result": result,
            "misc": misc,
            "exp_key": None,
            "owner": None,
            "version": 0,
            "book_time": datetime.datetime.now(),
            "refresh_time": datetime.datetime.now(),
        }
        docs.append(doc)

    return trials_from_docs(
        docs)  # built-in func to gen Trials for hyperopt from docs
