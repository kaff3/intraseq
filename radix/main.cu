
// Include the three versions of radix we want to test
#include"./radix.cuh"
#include"./radix-cub.cuh"
#include"../shared/helper.cu.h"
// Standard includes
#include<stdio.h>
#include<stdint.h>
#include<vector>
#include<sys/time.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<sstream>

// Cuda includes
#include<cuda_runtime.h>
#include<cub/cub.cuh>


int GetMask(int b){
    return (1 << b) - 1;
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


// Generic function to test the radix sort implementation
// arg N is the size of the array to be generated and sorted
template<
    typename T,
    int B,
    int E,
    int TS
    >
void test(size_t N, int runs) {

    size_t alloc_size = sizeof(T) * N;

    // Allocate host memory
    T* h_in  = (T*)malloc(alloc_size);
    T* h_out = (T*)malloc(alloc_size);
    randomInitNat(h_in, N, N);

    // Allocate device memory
    T* d_in;
    T* d_out;
    cudaMalloc((void**)&d_in, alloc_size);
    cudaMalloc((void**)&d_out, alloc_size);

    // Specialize instance of radix sort
    typedef Radix<T, B, E, TS> Radix;

    // Allocate auxillary storage needed 
    unsigned int* d_hist1;
    unsigned int* d_hist2;
    unsigned int* d_hist3;
    void* d_temp_storage;
    cudaMalloc((void**)&d_hist1, Radix::HistogramStorageSize(N));
    cudaMalloc((void**)&d_hist2, Radix::HistogramStorageSize(N));
    cudaMalloc((void**)&d_hist3, Radix::HistogramStorageSize(N));
    cudaMalloc((void**)&d_temp_storage, Radix::TempStorageSize(N, d_hist1));

    // Initialize array and move to device
    cudaMemcpy(d_in, h_in, alloc_size, cudaMemcpyHostToDevice);

    Radix::Sort(d_in, d_out, N, d_hist1, d_hist2, d_hist3, d_temp_storage, GetMask(B));
    cudaDeviceSynchronize();

    // Perform multiple runs
    Timer t1;
    t1.Start();
    for (int j = 0; j < runs; j++) {
        Radix::Sort(d_in, d_out, N, d_hist1, d_hist2, d_hist3, d_temp_storage, GetMask(B));
        cudaDeviceSynchronize();
    }
    t1.Stop();

    // Read out array to make sure nothing gets optimized away
    cudaMemcpy(h_out, d_out, alloc_size, cudaMemcpyDeviceToHost);

    #ifdef DO_VALIDATE 
        // Run the cub version for validation

        // Allocate cub memory
        T* h_out_cub = (T*)malloc(alloc_size);

        const unsigned int TILE_ELEMENTS = TS * E;
        size_t num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;

        // Prepare and execute
        cudaMemcpy(d_in, h_in, alloc_size, cudaMemcpyHostToDevice);
        RadixIntraCub<T, E, TS><<<num_blocks, TS>>>(d_in, d_out, N);
        cudaMemcpy(h_out_cub, d_out, alloc_size, cudaMemcpyDeviceToHost);


        bool valid = true;
        for (size_t k = 0; k < N; k++) {
            if (h_out[k] != h_out_cub[k]) {
                printf("Validation error at k = %lu\n", (unsigned long)k);
                printf("%lu != %lu\n", (unsigned long)h_out[k], (unsigned long)h_out_cub[k]);
                valid = false;
                break;
            }
        }
        if (valid)
            printf("Valid for N = %lu\n", (unsigned long)N);

        free(h_out_cub);
        // cudaFree(d_tmp);
    #endif

    #ifndef DO_VALIDATE
        printf("%lu, %.2f\n", (unsigned long)N, t1.Get());
    #endif

    // Free all memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist1);
    cudaFree(d_hist2);
    cudaFree(d_hist3);
    cudaFree(d_temp_storage);
}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Usage: %s <gpu runs>\n", argv[0]);
        return 0;
    }

    int gpu_runs = atoi(argv[1]);

  
    unsigned int size = 134217728;

    #ifdef DO_VALIDATE 
    size = 100000;
    #endif

    printf("Seq factor ======\n");
    test<unsigned int, 4, 1, 256>(size, gpu_runs); 
    test<unsigned int, 4, 4, 256>(size, gpu_runs); 
    test<unsigned int, 4, 8, 256>(size, gpu_runs); 
    test<unsigned int, 4, 22, 256>(size, gpu_runs); 

    printf("=====================================");
    test<unsigned int, 4, 4, 128>(size, gpu_runs); 
    test<unsigned int, 4, 4, 256>(size, gpu_runs); 
    test<unsigned int, 4, 4, 512>(size, gpu_runs); 
    test<unsigned int, 4, 4, 1024>(size, gpu_runs); 




    return 0;
}





