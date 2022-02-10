import pandas as pd

from hyperopt import hp
from hyperopt.base import Trials
import ax
from ax import Experiment, RangeParameter


def search_space_from_ax_experiment(exp: Experiment) -> dict:
    """
    convert ax-platform's experiment search space
        -> hyperopt's search space
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


def build_trials_from_sobol(exp: Experiment) -> Trials:

    # retrieve sobol
    df = exp.fetch_data().df.set_index("arm_name")
    arms_df = pd.DataFrame.from_dict(
        {k: v.parameters
         for k, v in exp.arms_by_name.items()}, orient="index")

    print(df)
    print(arms_df)

    # build trials TODO
    t = Trials()

    return t
