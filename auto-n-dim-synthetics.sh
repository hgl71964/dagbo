#! /bin/bash

# config
n=20
workload="rosenbrock-${n}D" # workload name need to match EXACTLY, it is lower case
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
        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_sobol.py \
                --n_dim $n \
                --exp_name ${exp_name}

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --n_dim $n \
                --performance_model_path dagbo/interface/rosenbrock_20d.txt \
                --tuner bo \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --n_dim $n \
                --performance_model_path dagbo/interface/rosenbrock_20d.txt \
                --tuner bo \
                --acq_name qEI

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --n_dim $n \
                --dagbo_mode direct \
                --performance_model_path dagbo/interface/rosenbrock_20d.txt \
                --tuner dagbo \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --n_dim $n \
                --dagbo_mode direct \
                --performance_model_path dagbo/interface/rosenbrock_20d.txt \
                --tuner dagbo \
                --acq_name qEI
done


