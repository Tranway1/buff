import re
import sys
import csv
onheap = []
skip = []
a = 0
b = 0
infile = sys.argv[1]
outfile = sys.argv[2]
time = int(sys.argv[3])

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# infile = "input.txt"
# outfile = "output.txt"
# time =3
cur = ""
i = 0
cr = 0.0
compthr = 0.0
decthr = 0.0
filterthr = 0.0
eqthr =0.0
sum = 0.0
max = 0.0
with open(outfile, 'w+') as the_file:
    with open(infile) as f:
        content = f.readlines()
        for line in content:
            if "ARGS" in line.rstrip().split(":")[0]:
                the_file.write("%s\n" % line.rstrip().split(":")[1])
