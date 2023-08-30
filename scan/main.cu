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
    cudaMalloc((void**)&d_in, REDUCE_TEST_SIZE * sizeof(int));
    cudaMalloc((void**)&d_out, num_blocks * sizeof(int));

    printf("Copying to device\n");
    cudaMemcpy(d_in, (void*)&foo, REDUCE_TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    printf("Running kernel\n");
    block_reduce<int, REDUCE_TEST_BLOCK_SIZE><<<num_blocks, REDUCE_TEST_BLOCK_SIZE>>>(d_in, d_out, REDUCE_TEST_SIZE);

    printf("Copying to host\n");
    cudaMemcpy((void*)&foo, d_out, REDUCE_TEST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Running loop \n");
    int sum = 0;
    for (int i = 0; i < num_blocks; i++) {
        sum += foo[i];
    }

    cudaDeviceSynchronize();
    printf("Reduced: %x\n", sum);
}

int main(int argc, char* argv[]) {
    testReduce();
    return 0;
}