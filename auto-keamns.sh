#! /bin/bash

# config
epochs=30
repeat=2
norm=1
minimize=1
device="cpu"

## hibench
workload="kmeans"
hibench_home="/home/gh512/workspace/bo/spark-dir/hiBench"
hdfs_path="/HiBench"

## paths
exec_path="/home/gh512/workspace/bo/spark-dir/hiBench/bin/workloads/ml/kmeans/spark/run.sh"
performance_model_path="dagbo/interface/spark_perf_model_8.txt"
hibench_report_path="/home/gh512/workspace/bo/spark-dir/hiBench/report/hibench.report"
conf_path="/home/gh512/workspace/bo/spark-dir/hiBench/conf/spark.conf"
base_url="http://localhost:18080"

clear_benchlog () {
  rm -rf /home/gh512/workspace/bo/spark-dir/hiBench/report/kmeans/spark
}

# create if not exist
mkdir -p benchmarks/data

## can add other python script positional args here

echo
echo "workload is: ${workload}"

# clean hdfs, will erase all hibench data
# NOTE: path like: /local/scratch/opt/hdfs_storage_dir/HiBench
hdfs dfs -rm -R ${hdfs_path}

set -e # stop on any error from now on

# run prepare script to gen data in hdfs
${hibench_home}/bin/workloads/ml/kmeans/prepare/prepare.sh

echo
echo "finish preparation: "
echo $(hdfs dfs -ls ${hdfs_path})
echo

for num in $( seq 0 $repeat )
do
        exp_name=${workload}-${num}
        echo
        echo "start experiment ${exp_name}: "
        echo

        # sequential experiment
        clear_benchlog
        python3.9 ./benchmarks/spark_continuous_modelling/spark_sobol.py \
                --bootstrap 5 \
                --exp_name ${exp_name} \
                --seed $num \
                --minimize $minimize \
                --conf_path ${conf_path} \
                --exec_path ${exec_path} \
                --hibench_report_path ${hibench_report_path} \
                --base_url ${base_url}

        clear_benchlog
        python3.9 ./benchmarks/spark_continuous_modelling/spark_continuation.py \
                --tuner dagbo-ssa \
                --acq_name qUCB \
                --epochs $epochs \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --seed $num \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path} \
                --conf_path ${conf_path} \
                --exec_path ${exec_path} \
                --hibench_report_path ${hibench_report_path} \
                --base_url ${base_url}

        clear_benchlog
        python3.9 ./benchmarks/spark_continuous_modelling/spark_continuation.py \
                --tuner bo \
                --acq_name qUCB \
                --epochs $epochs \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --seed $num \
                --norm $norm \
                --minimize $minimize \
                --device ${device} \
                --performance_model_path ${performance_model_path} \
                --conf_path ${conf_path} \
                --exec_path ${exec_path} \
                --hibench_report_path ${hibench_report_path} \
                --base_url ${base_url}

        clear_benchlog
        python3.9 ./benchmarks/spark_continuous_modelling/spark_hyperopt.py \
                --tuner rand \
                --epochs $epochs \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --seed $num \
                --minimize $minimize \
                --performance_model_path ${performance_model_path} \
                --conf_path ${conf_path} \
                --exec_path ${exec_path} \
                --hibench_report_path ${hibench_report_path} \
                --base_url ${base_url}

        clear_benchlog
        python3.9 ./benchmarks/spark_continuous_modelling/spark_hyperopt.py \
                --tuner tpe \
                --epochs $epochs \
                --load_name SOBOL-${exp_name} \
                --exp_name ${exp_name} \
                --seed $num \
                --minimize $minimize \
                --performance_model_path ${performance_model_path} \
                --conf_path ${conf_path} \
                --exec_path ${exec_path} \
                --hibench_report_path ${hibench_report_path} \
                --base_url ${base_url}
done


