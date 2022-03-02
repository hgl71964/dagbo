#! /bin/bash

# config
workload="rosenbrock" # workload name need to match EXACTLY, it is lower case
repeat=4

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
        exp_name=${workload}-${num}
        echo
        echo "start experiment ${exp_name}: "
        echo

        # sequential experiment
        python3.9 ./benchmarks/synthetic/rosenbrock_sobol.py \
                --exp_name ${exp_name}

        python3.9 ./benchmarks/synthetic/rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --performance_model_path dagbo/interface/rosenbrock_3d_correct_model.txt \
                --tuner bo \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --performance_model_path dagbo/interface/rosenbrock_3d_correct_model.txt \
                --tuner bo \
                --acq_name qEI

        python3.9 ./benchmarks/synthetic/rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --performance_model_path dagbo/interface/rosenbrock_3d_correct_model.txt \
                --tuner dagbo \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/rosenbrock_continuation_direct.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name}-direct \
                --performance_model_path dagbo/interface/rosenbrock_3d_correct_model.txt \
                --tuner dagbo \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --performance_model_path dagbo/interface/rosenbrock_3d_correct_model.txt \
                --tuner dagbo \
                --acq_name qEI

        python3.9 ./benchmarks/synthetic/rosenbrock_continuation_direct.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name}-direct \
                --performance_model_path dagbo/interface/rosenbrock_3d_correct_model.txt \
                --tuner dagbo \
                --acq_name qEI
done


