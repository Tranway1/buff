import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt


onheap = []
skip = []
a = 0
b = 0
infile = sys.argv[1]
# outfile = sys.argv[2]
skip = int(sys.argv[2])

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# skip = 1
# infile = "/Users/chunwei/research/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k"
outfile = infile.split('/')[-1]
print(outfile)
x=[]
with open(infile) as f:
    content = f.readlines()
    i = 0

    for line in content:
        lst = [float(n) for n in line.split(',') if isfloat(n)]
        x.extend(lst[skip:])

print(min(x),max(x))
matplotlib.rcParams.update({'font.size': 35})
# the histogram of the data
n, bins, patches = plt.hist(x, bins=100, weights=np.zeros_like(x) + 1. / len(x), facecolor='g', alpha=0.75)


plt.xlabel('Range')
plt.ylabel('Frequecy')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(outfile+"_hist.pdf")
