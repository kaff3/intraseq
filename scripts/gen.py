import sys
import json
import os
import re

import numpy as np

def groupTests(tests):
    org = []
    seq = []
    for test in tests:
        if "org" in test:
            org.append(test)
        else:
            seq.append(test)
    return zip(org, seq)

def typeToBytes(tp):
    match tp:
        case "u8":
            return 1
        case "u16":
            return 2
        case "u32":
            return 4
        case "u64":
            return 8
        case "i8":
            return 1
        case "i16":
            return 2
        case "i32":
            return 4
        case "i64":
            return 8

# input is e.g. of type "[n][m]u32" or if multiple "[n][m]u32 [n][m]u32"
def getWritten(input, test):
    match test:
        case "map-map-simple.fut":
            return input
        case "map-scan-simple.fut":
            return input
        case "map-reduce-simple.fut":
            idx = input.index(']')
            outer = input[:idx+1]
            idx = input.index(']', idx+1)
            tp = input[idx+1:]
            return outer +tp
        case "map-scatter-simple.fut":
            tmp = input.split(' ')
            return tmp[0]
        case _:
            return input

# converts the type of an array into its size in bytes. 
# The input is assumed to be two dimension, e.g "[1000][2000]i32"
def arrayByteSize(tp):
    tmp = tp.split(']')
    dims = tmp[:-1]

    for i in range(len(dims)):
        dims[i] = re.sub(r"^\W+", "", dims[i])
        dims[i] = int(dims[i])
    
    prim_size = typeToBytes(tmp[-1])
    size = 1
    for s in dims:
        size *= s
    return size * prim_size


def getBytesTransfered(read, write):
    read = read.split(' ')
    r_size = 0
    for r in read:
        r_size += arrayByteSize(r)
    write = write.split(' ')
    w_size = 0
    for w in write:
        w_size += arrayByteSize(w)
    return r_size + w_size

def getBandwidth(size, time):
    return (size/time) / 1000

def makePlot(data, org_test, seq_test):
    test_name = os.path.basename(org_test)
    print("================= " + test_name + " =================")
    org_datasets = data[org_test]['datasets']
    seq_datasets = data[seq_test]['datasets']

    for (o_data, seq_data) in zip(org_datasets, seq_datasets):
        # read the actual runtimes and take an average
        o_arr = np.array(org_datasets[o_data]['runtimes'])
        s_arr = np.array(seq_datasets[o_data]['runtimes'])

        read = o_data
        write = getWritten(read, test_name.split(':')[0])

        bytes_read_write = getBytesTransfered(read, write)

        print("read: " + read)
        print("write: " + write)

        print(f"bytes: {bytes_read_write}")

        o_avg = np.mean(o_arr)
        s_avg = np.mean(s_arr)
        print(o_avg)
        print(s_avg)

        o_gbs = getBandwidth(bytes_read_write, o_avg)
        s_gbs = getBandwidth(bytes_read_write, s_avg)

        print("GB/s:")
        print(f"   org: {o_gbs}")
        print(f"   seq: {s_gbs}")
        

        
        print("===============")




if len(sys.argv) != 2:
    print("Error: Expected two arguments")
    exit()

json_path = sys.argv[1]
if not os.path.exists(json_path):
    print("Error: " + json_path + " does not exist")
    exit()
    
print("readig json from: " + json_path)
with open(json_path, 'r') as file:
    data = json.load(file)

tests = groupTests(data.keys())

for org_test, seq_test in tests:
    makePlot(data, org_test, seq_test)
