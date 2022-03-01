#! /bin/bash

# config
workload="wordcount" # workload name need to match EXACTLY, it is lower case
hibench_home="/home/gh512/workspace/bo/spark-dir/hiBench"
#hdfs_path="/local/scratch/opt/hdfs_storage_dir/HiBench"
hdfs_path="/HiBench"
repeat=2

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
${hibench_home}/bin/workloads/micro/${workload}/prepare/prepare.sh

echo
echo "finish preparation: "
echo $(hdfs dfs -ls ${hdfs_path})
echo

for num in $( seq 0 $repeat )
do
        sobol_name=${workload}${num}
        exp_name=SOBOL-${sobol_name}
        echo
        echo "start experiment ${sobol_name}: "
        echo

        # sequential experiment
        python3.9 ./benchmarks/spark_sobol.py --exp_name ${sobol_name}
        python3.9 ./benchmarks/spark_continuation.py --exp_name ${exp_name} --tuner bo --acq_name qUCB
        python3.9 ./benchmarks/spark_hyperopt.py --exp_name ${exp_name} --tuner tpe
done


