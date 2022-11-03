CUB=cub-1.8.0

LINKS= -lcuda -lcudart -lnvrtc

all: cub-sort radix

cub-sort: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-cub sorting_test.cu
	./test-cub 100000000

radix: main.cu
	nvcc main.cu -o bin/radix

fut-bench: radix-fut.fut
	futhark bench --backend=cuda radix-fut.fut

clean:
	rm -f test-cub radix

