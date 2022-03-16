# parse the performance model from graphviz's source file
"""
# IMPORTANT: the graphviz's model is assumed to have 3 subgraphs in ORDER
  1. parameter space
  2. monitoring metrics
  3. objectives
"""


def _population_spaces(path: str):
    c = -1
    param_space = {}  # key: param name - val: continuous or categorical var
    metric_space = {}  # key: param name
    obj_space = {}  # key: param name
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split()
            if "subgraph" in l:
                c = int(l[1][-1])
                continue

            elif "}" in l:
                c = -1
                continue

            if c == 0:  # param
                if '=' not in l[0]:  # ok
                    #print(l[0].strip("\""))
                    param_name = l[0].strip("\"")
                    try:
                        shape = l[2]
                    except:
                        shape = None
                    ppt = "categorical"
                    if shape and "circle" in shape:
                        ppt = "continuous"
                    param_space[param_name] = ppt

            elif c == 1:  # metric
                if '=' not in l[0]:  # ok
                    name = l[0].strip("\"")
                    try:
                        shape = l[2]
                    except:
                        shape = None
                    ppt = 0
                    if shape and "square" in shape:
                        ppt = "add"
                    metric_space[name] = ppt

            elif c == 2:  # obj
                if '=' not in l[0]:  # ok
                    name = l[0].strip("\"")
                    try:
                        shape = l[2]
                    except:
                        shape = None
                    ppt = 0
                    if shape and "square" in shape:
                        ppt = "add"
                    obj_space[name] = ppt

    #print(param_space)
    #print(metric_space)
    #print(obj_space)
    return param_space, metric_space, obj_space


def parse_model(path):
    param_space, metric_space, obj_space = _population_spaces(path)
    edges = {}  # from key to val

    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split()
            if len(l) >= 3 and l[1] == '->':
                src = l[0].strip("\"")
                dst = l[2].strip("\"")

                if src in edges:
                    edges[src].append(dst)
                else:
                    edges[src] = [dst]

    #print(edges)
    #print(type(edges))
    return param_space, metric_space, obj_space, edges


if __name__ == '__main__':
    path = 'dagbo/interface/spark_performance_model.txt'
    parse_model(path)
