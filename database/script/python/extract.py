import re
import sys
import csv
import math
onheap = []
skip = []
a = 0
b = 0
infile = sys.argv[1]
outfile = sys.argv[2]
col = int(sys.argv[3])

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# col = 1
# infile = "input.txt"
# outfile = "output.txt"
with open(outfile, 'w+') as the_file:
  with open(infile) as f:
    content = f.readlines()
    i = 0
    for line in content:
      if not isfloat(line.rstrip().split(",")[col]):
        continue
      if math.isnan(float(line.rstrip().split(",")[col])):
        continue
      target = float(line.rstrip().split(",")[col])
      if (1):
        if (i==999):
          i=0
          the_file.write("%s\n" % (line.rstrip().split(",")[col]))
        else:
          the_file.write("%s," % (line.rstrip().split(",")[col]))
        i=i+1

