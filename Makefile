CUB=cub-1.8.0

LINKS= -lcuda -lcudart -lnvrtc

all: cub-sort radix

cub-sort: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-cub sorting_test.cu
	./test-cub 100000000

radix: radix-fut main.cu
	nvcc $(LINKS) main.cu radix-fut.o -o bin/radix

radix-main: main.cu
	nvcc main.cu -o bin/radix-main

radix-fut: radix-fut.fut
	futhark cuda --library radix-fut.fut -o radix-fut
	gcc radix-fut.c -c $(LINKS) -std=c99 -o radix-fut.o

# radix-cub:

clean:
	rm -f test-cub radix

