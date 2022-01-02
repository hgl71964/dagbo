import subprocess

rc = subprocess.call("benchmarks/sleep.sh", shell=True)
print(rc)

rc = subprocess.run(["ls", "-l"])
print(rc)
print("...")

rc = subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
print(rc)

