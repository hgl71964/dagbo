import ax
import json
import os
from os.path import join
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.decoder import experiment_from_json
from ax.storage.json_store.decoder import simple_experiment_from_json

if __name__ == "__main__":
    file_name = "spark_example_exp.json"
    srcdir = "."
    full_path = os.path.abspath(os.path.join(srcdir, file_name))
    print(full_path)

    # XXX simple experiment is deprecated and have error to load
    exp = simple_experiment_from_json(full_path)
    print(exp)

    #with open(full_path) as json_file:
    #    data = json.load(json_file)
    #    print(len(data))
    #    for k,v in data.items():
    #        print(k)
    #        print(v)
    #exp = load_experiment(full_path)
    #print(exp)
