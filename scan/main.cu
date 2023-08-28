#include<stdio.h>
#include<stdlib.h>
#include"./scan.cuh"
#include<cuda_runtime.h>
#include<algorithm>
#include<iterator>

#define REDUCE_TEST_SIZE 131072
#define REDUCE_TEST_BLOCK_SIZE 256

void testReduce() {
    int foo[REDUCE_TEST_SIZE];
    std::fill(std::begin(foo), std::end(foo), 1);
    int num_blocks = REDUCE_TEST_SIZE / REDUCE_TEST_BLOCK_SIZE;
    
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in, REDUCE_TEST_SIZE);
    cudaMalloc((void**)&d_out, num_blocks);

    block_reduce<int, REDUCE_TEST_BLOCK_SIZE><<<num_blocks, REDUCE_TEST_BLOCK_SIZE>>>(d_in, d_out, REDUCE_TEST_SIZE);

    int sum = 0;
    for (int i = 0; i < num_blocks; i++) {
        sum += d_out[i];
    }

    cudaDeviceSynchronize();
    printf("Reduced: %x\n", sum);
}

int main(int argc, char* argv[]) {
    testReduce();
    return 0;
}