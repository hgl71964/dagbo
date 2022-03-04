#! /bin/bash

# config
n=3
epochs=30
workload="rosenbrock-${n}D" # workload name need to match EXACTLY, it is lower case
repeat=4
norm=1

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
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --performance_model_path dagbo/interface/rosenbrock_3d_dagbo.txt \
                --tuner bo \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --performance_model_path dagbo/interface/rosenbrock_3d_dagbo.txt \
                --tuner bo \
                --acq_name qEI

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --performance_model_path dagbo/interface/rosenbrock_3d_dagbo.txt \
                --tuner dagbo-ssa \
                --acq_name qUCB

        python3.9 ./benchmarks/synthetic/n_dim_rosenbrock_continuation.py \
                --seed $num \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --epochs $epochs \
                --n_dim $n \
                --norm $norm \
                --performance_model_path dagbo/interface/rosenbrock_3d_dagbo.txt \
                --tuner dagbo-ssa \
                --acq_name qEI
done


