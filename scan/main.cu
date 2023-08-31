#include<stdio.h>
#include<stdlib.h>
#include"./scan.cuh"
#include"../timing.h"
#include<cuda_runtime.h>
#include<algorithm>
#include<iterator>

#define TEST_SIZE (1024 * 1500)
#define BLOCK_SIZE 1024


// void testReduce() {
//     int foo[REDUCE_TEST_SIZE];
//     std::fill(std::begin(foo), std::end(foo), 1);
//     int num_blocks = (REDUCE_TEST_SIZE + REDUCE_TEST_BLOCK_SIZE - 1) / REDUCE_TEST_BLOCK_SIZE;
//     int shmem_size = REDUCE_TEST_BLOCK_SIZE * sizeof(int);

//     printf("num blocks = %i\n", num_blocks);
//     printf("shmem size = %i bytes\n", shmem_size);
    
//     int* d_in;
//     int* d_out;
//     cudaMalloc((void**)&d_in, REDUCE_TEST_SIZE * sizeof(int));
//     cudaMalloc((void**)&d_out, num_blocks * sizeof(int));

//     printf("Copying to device\n");
//     cudaMemcpy(d_in, (void*)&foo, REDUCE_TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);

//     printf("Running kernel\n");
//     reduce_kernel<int><<<num_blocks, REDUCE_TEST_BLOCK_SIZE, shmem_size>>>(d_in, d_out, REDUCE_TEST_SIZE);

//     printf("Copying to host\n");
//     cudaMemcpy((void*)&foo, d_out, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

//     printf("Running loop \n");
//     int sum = 0;
//     for (int i = 0; i < num_blocks; i++) {
//         sum += foo[i];
//     }

//     cudaDeviceSynchronize();
//     printf("Reduced: %i\n", sum);

//     cudaFree((void*)d_in);
//     cudaFree((void*)d_out);
// }


template<typename T>
void intraBlockScanBench() {
    int num_blocks = TEST_SIZE / BLOCK_SIZE;

    // Create the array to be scanned
    T* arr1 = (T*)malloc(TEST_SIZE*sizeof(T));
    T* arr2 = (T*)malloc(TEST_SIZE*sizeof(T));
    // std::fill(std::begin(arr1), std::end(arr1), 1);
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
    // scan_kernel<T><<<num_blocks, BLOCK_SIZE>>>(d_in1, TEST_SIZE, 1000);
    // cudaDeviceSynchronize();
    // cudaMemcpy(d_in1, (void*)&arr1, TEST_SIZE*sizeof(T), cudaMemcpyHostToDevice);

    // Benchamrking
    Timer t1, t2;
    
    const size_t iterations = 1000000;

    t1.Start();
    scan_kernel<T><<<num_blocks, BLOCK_SIZE>>>(d_in1, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t1.Stop();

    t2.Start();
    scan_kernel_seq<T><<<num_blocks, BLOCK_SIZE/4>>>(d_in2, TEST_SIZE, iterations);
    cudaDeviceSynchronize();
    t2.Stop();

    printf("nor = %.2f\n", t1.Get());
    printf("seq = %.2f\n", t2.Get());

    cudaMemcpy((void*)arr1, d_in1, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)arr2, d_in2, TEST_SIZE*sizeof(T), cudaMemcpyDeviceToHost);

    printf("loop starting\n");
    bool succes = true;
    for (size_t i = 0; i < TEST_SIZE; i++) {
       if (arr1[i] != arr2[i]) {
           succes = false;
           printf("oh no at i=%i\n", i);
           break;
       }
    }

    printf("succes = %i\n", succes);

    cudaFree(d_in1);
    cudaFree(d_in2);
}



int main(int argc, char* argv[]) {
    intraBlockScanBench<unsigned int>();
    return 0;
}