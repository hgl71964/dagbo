#! /bin/bash

# config
epochs=50
workload="hartmann-6D" # workload name need to match EXACTLY, it is lower case
repeat=9
norm=1
minimize=1
device=gpu

performance_model_path="dagbo/interface/hartmann_perf_model_2.txt"
# create if not exist
mkdir -p benchmarks/data

## can add other python script positional args here

echo
echo "workload is: ${workload}"

set -e # stop on any error from now on

for num in $( seq 0 $repeat )
do
        exp_name=${workload}-${num}
        echo
        echo "start experiment ${exp_name}: "
        echo

        # sequential experiment
        python3.9 ./benchmarks/synthetic/hartmann_sobol.py \
                --seed $num \
                --exp_name ${exp_name}

        python3.9 ./benchmarks/synthetic/hartmann_continuation.py \
                --tuner dagbo-ssa \
                --acq_name qUCB \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}

        python3.9 ./benchmarks/synthetic/hartmann_continuation.py \
                --tuner dagbo-direct \
                --acq_name qUCB \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}


        python3.9 ./benchmarks/synthetic/hartmann_continuation.py \
                --tuner bo \
                --acq_name qUCB \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}
done
