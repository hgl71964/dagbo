#! /bin/bash

# config
n=3
epochs=30
workload="rosenbrock-${n}D" # workload name need to match EXACTLY, it is lower case
repeat=9
norm=1
minimize=1  # rosenbrock obj is to be minimized
device=gpu

performance_model_path="dagbo/interface/rosenbrock_3d_wrong_dagbo.txt"
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
                --seed $num \
                --n_dim $n \
                --exp_name ${exp_name}

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --tuner dagbo-ssa \
                --acq_name qUCB \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --tuner dagbo-ssa \
                --acq_name qEI \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --tuner bo \
                --acq_name qUCB \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --tuner bo \
                --acq_name qEI \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path}
done
