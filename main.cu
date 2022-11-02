
// Include the three versions of radix we want to test
#include "./radix.cuh"
#include "./radix-cub.cuh"
// #include "radix-fut.h"  // automagically generated
#include "./helper.cu.h"
// Standard includes
#include<stdio.h>
#include<stdint.h>
#include<vector>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>


// Cuda includes
#include"cub/cub.cuh"
#include<cuda_runtime.h>

template<typename T>
void randomInitNat(T* data, const size_t size, const size_t H) {
    for (size_t i = 0; i < size; i++) {
        T r = rand();
        data[i] = r % H;
    }
}

template<typename T>
bool validate(T* h1, T* h2, int N) {
    bool valid = true;
    for (int i = 0; i < N; i++) {
        if (h1[i] != h2[i]) {
            valid = false;
            break;
        }
    }
    return valid;
}

template<typename T>
void bench(std::vector<int> sizes) {
    
    for (int i = 0; i < sizes.size(); i++) {
        int N = sizes[i];
        size_t arr_size = N * sizeof(T);

        // Host allocations
        T* h_in      = (T*)malloc(arr_size);
        T* h_out_our = (T*)malloc(arr_size);
        T* h_out_cub = (T*)malloc(arr_size);
        T* h_out_fut = (T*)malloc(arr_size);

        // Instantiate our radix sort algorithm with template with a typedef
        typedef Radix<T, 4, 4, 256> Radix4;

        // Device allocations
        T* d_in;
        T* d_out;
        unsigned int* d_histogram;
        cudaMalloc((void**)&d_in,  arr_size);
        cudaMalloc((void**)&d_out, arr_size);
        cudaMalloc((void**)&d_histogram, Radix4::d_histogramSize(N));

        // Initialize the array to be sorted and transfer to device
        randomInitNat<T>(h_in, N, N);
        cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

        // Run our version and save the result
        double elapsed_us;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        Radix4::Sort(d_in, d_out, N, d_histogram, 0xF);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_us = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("Our:    %i in   %.2f\n", sizes[i],elapsed_us);

        cudaMemcpy(h_out_our, d_in, arr_size, cudaMemcpyDeviceToHost);

        // Now the CUB version
        cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

        double elapsed_cub;
        struct timeval t_start_cub, t_end_cub, t_diff_cub;
        gettimeofday(&t_start_cub, NULL);

        RadixSortCub<T>(d_in, d_out, N);

        gettimeofday(&t_end_cub, NULL);
        timeval_subtract(&t_diff_cub, &t_end_cub, &t_start_cub);
        elapsed_cub = (t_diff_cub.tv_sec*1e6+t_diff_cub.tv_usec);
        printf("Cub:    %i in   %.2f\n", sizes[i],elapsed_cub);

        cudaMemcpy(h_out_cub, d_out, arr_size, cudaMemcpyDeviceToHost);

        // Validate if our implementation did it correct
        printf("Validation: ");
        if (validate<T>(h_out_our, h_out_cub, N)) {
            printf("VALID\n");
        } else {
            printf("INVALID\n");
        }

        // for (int j = 0; j < N; j++) {
        //     printf("%10x   %10x   %10x\n", h_out_our[j], h_out_cub[j], h_in[j]);
        // }

        // Have to allocate and free eeach iteration as the sizes change
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_histogram);
        free(h_in);
        free(h_out_our);
        free(h_out_cub);
        free(h_out_fut);
    }
}


int main(int argc, char* argv[]) {

    // if (argc < 2) {
    //     printf("Usage: ./radix <array size>\n");
    //     return 0;
    // }

    // int N = atoi(argv[1]);
    // size_t arr_size = N * sizeof(unsigned int);

    std::vector<int> sizes;
    sizes.push_back(333);       
    sizes.push_back(1024);       
    sizes.push_back(1000000);
    sizes.push_back(10000000);
    sizes.push_back(100000000);

    printf("Unsigned int:\n");
    bench<unsigned int>(sizes);

    printf("Unsigned long:\n");
    bench<unsigned long>(sizes);

    printf("Unsigned short:\n");
    bench<unsigned short>(sizes);
    
    printf("Unsigned char:\n");
    bench<unsigned char>(sizes);


    return 0;
}





