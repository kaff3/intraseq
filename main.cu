
// Include the three versions of radix we want to test
#include "radix.cuh"
// #include "radix-fut.h"  // automagically generated

// Standard includes
#include<stdio.h>
#include<stdint.h>

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


int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Usage: ./radix <array size>\n");
        return 0;
    }

    int N = atoi(argv[1]);
    size_t arr_size = N * sizeof(unsigned int);

    // Instantiate. For easier calling of functions
    typedef Radix<unsigned int, 4, 4, 256> Radix4;

    unsigned int* h_in  = (unsigned int*)malloc(arr_size);
    unsigned int* h_out = (unsigned int*)malloc(arr_size);

    randomInitNat<unsigned int>(h_in, N, N);

    unsigned int* d_histogram;
    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc((void**)&d_in,  arr_size);
    cudaMalloc((void**)&d_out, arr_size);
    cudaMalloc((void**)&d_histogram, Radix4::d_histogramSize(N));

    // Copy initial array to device
    cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

    Radix4::Sort(d_in, d_out, N, d_histogram, 0xF);

    cudaMemcpy(h_out, d_in, arr_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%10x      %10x\n", h_out[i], h_in[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_histogram);
    free(h_in);
    free(h_out);

    return 0;
}





