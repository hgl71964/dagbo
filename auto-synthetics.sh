#! /bin/bash

# config
workload="rosenbrock" # workload name need to match EXACTLY, it is lower case
repeat=2

# create if not exist
mkdir -p benchmarks/data

## can add other python script positional args here

echo
echo "workload is: ${workload}"

set -e # stop on any error from now on

echo
echo "finish preparation: "
echo

for num in $( seq 0 $repeat )
do
        sobol_name=${workload}${num}
        exp_name=SOBOL-${sobol_name}
        echo
        echo "start experiment ${sobol_name}: "
        echo

        # sequential experiment
        python3.9 ./benchmarks/synthetic/rosenbrock_sobol.py \
                --exp_name ${sobol_name}
        python3.9 ./benchmarks/synthetic/rosenbrock_continuation.py \
                --exp_name ${exp_name} \
                --tuner bo \
                --acq_name qUCB
        python3.9 ./benchmarks/synthetic/rosenbrock_continuation.py \
                --exp_name ${exp_name} \
                --tuner dagbo \
                --acq_name qUCB
        python3.9 ./benchmarks/synthetic/rosenbrock_continuation_direct.py \
                --exp_name ${exp_name} \
                --tuner dagbo \
                --acq_name qUCB
done


