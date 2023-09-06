import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = sys.argv[1]

data = np.genfromtxt(path, delimiter=",").transpose()

# First graph
plt.scatter(data[0], data[1], label="Sequentialized")
plt.plot(data[0], data[1])
plt.scatter(data[0], data[2], label="Shared memory")
plt.plot(data[0], data[2])
plt.scatter(data[0], data[3], label="Registers")
plt.plot(data[0], data[3])
plt.legend()
plt.xscale("log", base=2)
plt.title("Intra block scan runtimes")
plt.xlabel("Number of elements")
plt.ylabel("Runtime in ms?")


plt.savefig("tmp_runtimes.jpg")
plt.show()


# Second Graph
plt.scatter(data[0], data[4], label="Seq/sharedMem speedup")
plt.plot(data[0], data[4])
plt.scatter(data[0], data[5], label="sharedMem/registers speedup")
plt.plot(data[0], data[5])
plt.xscale("log", base=2)

plt.title("Intra block scan speedups")
plt.xlabel("Number of elements")
plt.ylabel("Speedup")

plt.savefig("tmp_speedups.jpg")
plt.show()

A