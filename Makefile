CUB=cub-1.8.0

all: cub-sort radix

cub-sort: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-cub sorting_test.cu
	./test-cub 100000000

radix: radix-fut main.cu
	nvcc main.cu baseline.c -o bin/radix

radix-main: main.cu
	nvcc main.cu -o bin/radix-main

radix-fut: baseline.fut
	futhark cuda --library radix-fut.fut

# radix-cub:

clean:
	rm -f test-cub radix

