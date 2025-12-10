import depthai as dai
import sys

print("1. Path to depthai:", dai.__file__)
print("2. Version:", dai.__version__)
print("3. Available attributes in dai.node:")

try:
    print(dir(dai.node))
except Exception as e:
    print(f"ERROR: Could not inspect dai.node. Details: {e}")import depthai as dai
import sys

print("1. Path to depthai:", dai.__file__)
print("2. Version:", dai.__version__)
print("3. Available attributes in dai.node:")

try:
    print(dir(dai.node))
except Exception as e:
    print(f"ERROR: Could not inspect dai.node. Details: {e}")