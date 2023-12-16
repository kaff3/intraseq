
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

// template<
//     typename T, 
//     int B, 
//     int E,
//     int TS >
// void bench(std::vector<size_t> sizes, int gpu_runs, const char* out_file) {
    
//     std::vector<float> avg_our;
//     std::vector<float> avg_cub;

//     for (int i = 0; i < sizes.size(); i++) {
//         size_t N = sizes[i];
//         printf("===============================\n");
//         printf("N: %lu\n", N);
//         size_t arr_size = N * sizeof(T);


//         // Host allocations
//         T* h_in      = (T*)malloc(arr_size);
//         T* h_out_our = (T*)malloc(arr_size);
//         T* h_out_cub = (T*)malloc(arr_size);

//         // Instantiate our radix sort algorithm with template with a typedef
//         typedef Radix<T, B, E, TS> Radix4;
//         int mask = GetMask(B);

//         // Device allocations
//         T* d_in;
//         T* d_out;
//         unsigned int* d_histogram1;
//         unsigned int* d_histogram2;
//         unsigned int* d_histogram3;
//         void*         d_tmp_storage;
//         cudaMalloc((void**)&d_in,  arr_size);
//         cudaMalloc((void**)&d_out, arr_size);
//         cudaMalloc((void**)&d_histogram1, Radix4::HistogramStorageSize(N));
//         cudaMalloc((void**)&d_histogram2, Radix4::HistogramStorageSize(N));
//         cudaMalloc((void**)&d_histogram3, Radix4::HistogramStorageSize(N));
//         cudaMalloc((void**)&d_tmp_storage, Radix4::TempStorageSize(N, d_histogram1));

//         // Allocations for cub version
//         void* d_tmp_storage_cub = NULL;
//         size_t tmp_storage_bytes = 0;
//         cub::DeviceRadixSort::SortKeys(d_tmp_storage_cub, tmp_storage_bytes, d_in, d_out, N);
//         cudaMalloc(&d_tmp_storage_cub, tmp_storage_bytes);


//         // Dry runs
//         Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
//         RadixSortCub<T>(d_in, d_out, N, d_tmp_storage_cub, tmp_storage_bytes);
//         cudaDeviceSynchronize();

//         std::vector<Timer> time_our;
//         std::vector<Timer> time_cub;

//         for (int j = 0; j < gpu_runs; j++) {

//             // Initialize the array to be sorted and transfer to device
//             randomInitNat<T>(h_in, N, N);

//             // Timers for our version and cub
//             Timer t1, t2;

//             // Move array to device
//             cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

//             // Run our version and save the result
//             t1.Start();
//             Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
//             cudaDeviceSynchronize();
//             t1.Stop();

//             #ifdef RADIX_VALIDATE
//             // Save sorted array to host for validation
//             cudaMemcpy(h_out_our, d_in, arr_size, cudaMemcpyDeviceToHost);
//             #endif

//             // Now the CUB version
//             cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);
 
//             t2.Start();
//             RadixSortCub<T>(d_in, d_out, N, d_tmp_storage_cub, tmp_storage_bytes);
//             cudaDeviceSynchronize();
//             t2.Stop();


//             #ifdef RADIX_VALIDATE
//             cudaMemcpy(h_out_cub, d_out, arr_size, cudaMemcpyDeviceToHost);
//             // Print if we do not validate
//             if (!validate<T>(h_out_our, h_out_cub, N)) {
//                 printf("INVALID. Size %i run %i\n", N, j);
//             }
//             #endif

//             // Save runtimes
//             time_our.push_back(t1);
//             time_cub.push_back(t2);
//         }

//         // Save the average runtimes
//         float run_our = average(time_our);
//         float run_cub = average(time_cub);

//         avg_our.push_back(run_our);
//         avg_cub.push_back(run_cub);

//         printf("Our: %.2f\n", run_our);
//         printf("Cub: %.2f\n", run_cub);
//         printf("factor: %f\n", run_our/run_cub);


//         // Have to allocate and free each iteration of outer loop as the sizes change but they are not timed
//         cudaFree(d_in);
//         cudaFree(d_out);
//         cudaFree(d_histogram1);
//         cudaFree(d_histogram2);
//         cudaFree(d_histogram3);
//         cudaFree(d_tmp_storage);
//         cudaFree(d_tmp_storage_cub);
//         free(h_in);
//         free(h_out_our);
//         free(h_out_cub);
//     }

//     writeRuntimes(sizes, avg_our, avg_cub, out_file);

// }

// template<
//     typename T, 
//     int B, 
//     int E,
//     int TS >
// void benchTuning(std::vector<size_t> sizes, int gpu_runs, const char* out_file) {

//     std::vector<float> avg_times;
    
//     for (int i = 0; i < sizes.size(); i++) {
//         size_t N = sizes[i];
//         printf("===============================\n");
//         printf("N: %lu\n", N);
//         size_t arr_size = N * sizeof(T);


//         // Host allocations
//         T* h_in      = (T*)malloc(arr_size);

//         // Instantiate our radix sort algorithm with template with a typedef
//         typedef Radix<T, B, E, TS> Radix4;
//         int mask = GetMask(B);

//         // Device allocations
//         T* d_in;
//         T* d_out;
//         unsigned int* d_histogram1;
//         unsigned int* d_histogram2;
//         unsigned int* d_histogram3;
//         void*         d_tmp_storage;
//         cudaMalloc((void**)&d_in,  arr_size);
//         cudaMalloc((void**)&d_out, arr_size);
//         cudaMalloc((void**)&d_histogram1, Radix4::HistogramStorageSize(N));
//         cudaMalloc((void**)&d_histogram2, Radix4::HistogramStorageSize(N));
//         cudaMalloc((void**)&d_histogram3, Radix4::HistogramStorageSize(N));
//         cudaMalloc((void**)&d_tmp_storage, Radix4::TempStorageSize(N, d_histogram1));

//         // Dry run
//         Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
//         cudaDeviceSynchronize();

//         std::vector<Timer> times;

//         for (int j = 0; j < gpu_runs; j++) {

//             Timer t1;

//              // Initialize the array to be sorted and transfer to device
//             randomInitNat<T>(h_in, N, N);

//             // Move array to device
//             cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

//             // Run our version and save the result
//             t1.Start();
//             Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
//             cudaDeviceSynchronize();
//             t1.Stop();

//             times.push_back(t1);
//         }

//         float avg = average(times);
//         avg_times.push_back(avg);

//         printf("Time: %.2f\n", avg);

//         cudaFree(d_in);
//         cudaFree(d_out);
//         cudaFree(d_histogram1);
//         cudaFree(d_histogram2);
//         cudaFree(d_histogram3);
//         cudaFree(d_tmp_storage);
//         free(h_in);
//     }

//     writeRuntimes(sizes, avg_times, avg_times, out_file);

// }

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

        void* d_tmp = NULL;
        size_t tmp_size_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_tmp, tmp_size_bytes, d_in, d_out, N);
        cudaMalloc((void**)&d_tmp, tmp_size_bytes);

        // Prepare and execute
        cudaMemcpy(d_in, h_in, alloc_size, cudaMemcpyHostToDevice);
        RadixSortCub<T>(d_in, d_out, N, d_tmp, tmp_size_bytes);
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
        cudaFree(d_tmp);
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

    std::vector<unsigned int> sizes = {
        100000,
        200000,
        300000,
        400000,
        500000
        // 1 << 10,
        // 1 << 11,
        // 1 << 12,
        // 1 << 13,
        // 1 << 14,
        // 1 << 15,
        // 1 << 16,
        // 1 << 17,
        // 1 << 19,
        // 1 << 20,
        // 1 << 21,
        // 1 << 22,
        // 1 << 23,
        // 1 << 24,
        // 1 << 25,
        // 1 << 26,
        // 1 << 27,
        // 1 << 28,
        // 1 << 29,
        // 1 << 30,
        // 1 << 31,
        // 1 << 32,
    };

    for(int i = 0; i < sizes.size(); i++) {
       test<unsigned int, 4, 4, 256>(sizes[i], gpu_runs); 
    } 

    printf("==================================================\n");
    for(int i = 0; i < sizes.size(); i++) {
        test<unsigned int, 4, 4, 1024>(sizes[i], gpu_runs);
    }



    return 0;
}





