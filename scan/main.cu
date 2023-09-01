#include<stdio.h>
#include<stdlib.h>
#include"./scan.cuh"
#include"../shared/timing.h"
#include<cuda_runtime.h>
#include<algorithm>
#include<iterator>

#define TEST_SIZE (1024 * 100)
#define BLOCK_SIZE 1024

template<typename T>
void intraBlockScanBench() {
    int num_blocks = TEST_SIZE / BLOCK_SIZE;

    // Create the array to be scanned
    T* arr1 = (T*)malloc(TEST_SIZE*sizeof(T));
    T* arr2 = (T*)malloc(TEST_SIZE*sizeof(T));
    for (size_t i = 0; i < TEST_SIZE; i++) {
        arr1[i] = 1;
    }


    // Create device memory and copy
    T* d_in1;
    T* d_in2;
    cudaMalloc((void**)&d_in1, TEST_SIZE*sizeof(T));
    cudaMalloc((void**)&d_in2, TEST_SIZE*sizeof(T));
    cudaMemcpy(d_in1, (void*)arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, (void*)arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);

    // Dry run
    //scan_kernel<T><<<num_blocks, BLOCK_SIZE>>>(d_in1, TEST_SIZE, 100);
    //cudaDeviceSynchronize();
    //cudaMemcpy(d_in1, (void*)&arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);

    // Benchmarking
    Timer t1, t2, t3;
    
    const size_t iterations = 10000;

    t1.Start();
    scan_kernel<T><<<num_blocks, BLOCK_SIZE>>>(d_in1, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t1.Stop();

    t2.Start();
    scan_kernel_seq<T><<<num_blocks, BLOCK_SIZE/4>>>(d_in2, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t2.Stop();

    //t3.Start();
    //scan_kernel_seq_reg<T><<<num_blocks, BLOCK_SIZE/4>>>(d_in2, TEST_SIZE, iterations);
    //cudaDeviceSynchronize();
    //t3.Stop();

    printf("nor = %.2f\n", t1.Get());
    printf("seq = %.2f\n", t2.Get());
    //printf("seqReg = %.2f\n", t3.Get());

    cudaMemcpy((void*)arr1, d_in1, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)arr2, d_in2, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);

    printf("loop starting\n");
    bool success = true;
    for (size_t i = 0; i < TEST_SIZE; i++) {
       if (arr1[i] != arr2[i]) {
           printf("Arr1: %u\n", arr1[i]);
           printf("Arr2: %u\n", arr2[i]);
           success = false;
           printf("oh no at i=%i\n", i);
           break;
       }
    }

    printf("success = %i\n", success);

    cudaFree(d_in1);
    cudaFree(d_in2);
}



int main(int argc, char* argv[]) {
    intraBlockScanBench<unsigned int>();
    return 0;
}
