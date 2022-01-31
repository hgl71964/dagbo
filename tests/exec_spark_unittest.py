import ax
import unittest
import time

from dagbo.interface.exec_spark import *

class test_exec_spark(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skip("ok")
    def test_subprocess(self):

        rc = subprocess.run(["ls", "-l"])
        print(rc)
        print("...")

        t = time.time()
        rc = subprocess.run(["sleep", "3"])
        print(rc)
        print(time.time() - t)

        if rc.returncode != 0:
            print("return != 0")

    @unittest.skip("ok, this will overwrite conf file")
    def test_write_spec(self):
        file_path = "/home/gh512/workspace/bo/spark-dir/hiBench/conf/spark.conf"
        exec_path = "/home/gh512/workspace/bo/spark-dir/hiBench/bin/workloads/micro/wordcount/spark/run.sh"
        spec = {
                "executor.num[*]": 2,
                "executor.cores": 1,
                "executor.memory": 2,
                "shuffle.compress": 0,
                }
        call_spark(spec, file_path, exec_path)



if __name__ == "__main__":
    unittest.main()
