#include<stdio.h>
#include<stdlib.h>
#include"./scan.cuh"
#include"../shared/timing.h"
#include<cuda_runtime.h>
#include<algorithm>
#include<iterator>
#include<vector>

// #define TEST_SIZE (1024 * 1000)
// #define BLOCK_SIZE 1024

template<typename T>
void intraBlockScanBench(const unsigned int block_size, 
                         const unsigned int num_blocks,
                         const unsigned int num_elems,
                         const unsigned int iterations) 
{
    
    // compute total number of elements
    const unsigned int TEST_SIZE = block_size * num_blocks;

    // Create the array to be scanned
    T* arr1 = (T*)malloc(TEST_SIZE*sizeof(T));
    T* arr2 = (T*)malloc(TEST_SIZE*sizeof(T));
    T* arr3 = (T*)malloc(TEST_SIZE*sizeof(T));
    for (size_t i = 0; i < TEST_SIZE; i++) {
        arr1[i] = 1;
    }


    // Create device memory and copy
    T* d_in1;
    T* d_in2;
    T* d_in3;
    cudaMalloc((void**)&d_in1, TEST_SIZE*sizeof(T));
    cudaMalloc((void**)&d_in2, TEST_SIZE*sizeof(T));
    cudaMalloc((void**)&d_in3, TEST_SIZE*sizeof(T));
    cudaMemcpy(d_in1, (void*)arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, (void*)arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in3, (void*)arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);

    // Dry run
    scan_kernel<T><<<num_blocks, block_size>>>(d_in1, TEST_SIZE, 100);
    cudaDeviceSynchronize();
    cudaMemcpy(d_in1, (void*)arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);

    // Benchmarking
    Timer t1, t2, t3;

    t1.Start();
    for(int k=0; k<GPU_RUNS; k++)
      scan_kernel<T><<<num_blocks, block_size>>>(d_in1, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t1.Stop();

    t2.Start();
    for(int k=0; k<GPU_RUNS; k++)
      scan_kernel_seq<T><<<num_blocks, block_size/num_elems>>>(d_in2, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t2.Stop();

    t3.Start();
    for(int k=0; k<GPU_RUNS; k++)
      scan_kernel_seq_reg<T><<<num_blocks, block_size/num_elems>>>(d_in3, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t3.Stop();

    

    cudaMemcpy((void*)arr1, d_in1, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)arr2, d_in2, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)arr3, d_in3, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);

    #ifdef DO_VALIDATE
    printf("loop starting\n");
    bool success = true;
    for (size_t i = 0; i < TEST_SIZE; i++) {
       if ( (arr1[i] != arr2[i]) || (arr2[i] != arr3[i])) {
           printf("Arr1: %u\n", arr1[i]);
           printf("Arr2: %u\n", arr2[i]);
           printf("Arr3: %u\n", arr3[i]);
           success = false;
           printf("oh no at i=%i\n", i);
           break;
       }
    printf("success = %i\n", success);
    }
    #endif

    // Print csv outpu
    unsigned int seq_spdup = t1.Get() / t2.Get();
    unsigned int reg_spdup = t1.Get() / t3.Get();
    printf("%u, %.2f, %.2f, %.2f, %.2f, %.2f\n", TEST_SIZE, t1.Get(), t2.Get(), t3.Get(), seq_spdup, reg_spdup);


    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_in3);
    free(arr1);
    free(arr2);
    free(arr3);
}



int main(int argc, char* argv[]) {
    const unsigned int iterations = 3;

    // The number of blocks to test with
    std::vector<unsigned int> num_blocks = {
        1 << 0,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
        1 << 8,
        1 << 9,
        1 << 10,
        // 1 << 11,
        // 1 << 12,
        // 1 << 13,
        // 1 << 14,
        // 1 << 15,
        // 1 << 16,
    };

    for (size_t i = 0; i < num_blocks.size(); i++) {
        intraBlockScanBench<unsigned int>(1024, num_blocks[i], 4, iterations);
    }
    return 0;
}