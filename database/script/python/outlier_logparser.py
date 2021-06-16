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
            if line.rstrip().split(":")[0] == "ARGS":
                cur= line.rstrip().split(":")[1]
            elif line.rstrip().split(":")[0] == "Number of qualified items":
                cur+= line.rstrip().split(":")[1]
                cur+=","
            elif line.rstrip().split(":")[0] == "Number of qualified items for equal":
                cur+= line.rstrip().split(":")[1]
                cur+=","
            elif line.rstrip().split(":")[0] == "Performance":
                i+=1
                cur+= line.rstrip().split(":")[1]
                arr = cur.rstrip().split(",")
                # print cur
                cr += float(arr[5])
                compthr += float(arr[6])
                decthr += float(arr[7])
                filterthr += float(arr[8])
                eqthr += float(arr[9])
                sum += float(arr[10])
                max += float(arr[11])
                # the_file.write("%s\n" % cur)
                if i==time:
                    paramters = arr[:4]
                    paramters.append(str(cr/time))
                    paramters.append(str(compthr/time))
                    paramters.append(str(decthr/time))
                    paramters.append(str(filterthr/time))
                    paramters.append(str(eqthr/time))
                    paramters.append(str(sum/time))
                    paramters.append(str(max/time))
                    overview =','.join([str(x) for x in paramters])
                    the_file.write("%s\n" % overview)
                    i = 0
                    cr = 0.0
                    compthr = 0.0
                    decthr = 0.0
                    filterthr = 0.0
                    eqthr = 0.0
                    sum = 0.0
                    max = 0.0
                cur=""
