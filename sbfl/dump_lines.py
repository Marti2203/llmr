import sys

path = sys.argv[1]
max = int(sys.argv[2].strip())
s = "\n".join([f"{path}:{i}" for i in range(max)])
print(s)