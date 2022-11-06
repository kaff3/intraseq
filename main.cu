
// Include the three versions of radix we want to test
// #include "./radix.cuh"
#include"./radix-no-opt.cuh"
#include"./radix-cub.cuh"
#include"./helper.cu.h"
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
#include"cub/cub.cuh"
#include<cuda_runtime.h>


int GetMask(int b){
    int res = 0;
    for (int i = 0; i < b; i++) {
        res = res << 1;
        res = res | 1;
    }
    
    return res;
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

template<
    typename T, 
    int B, 
    int E,
    int TS >
void bench(std::vector<size_t> sizes, int gpu_runs, const char* out_file) {
    
    std::vector<float> avg_our;
    std::vector<float> avg_cub;

    for (int i = 0; i < sizes.size(); i++) {
        size_t N = sizes[i];
        printf("===============================\n");
        printf("N: %lu\n", N);
        size_t arr_size = N * sizeof(T);


        // Host allocations
        T* h_in      = (T*)malloc(arr_size);
        T* h_out_our = (T*)malloc(arr_size);
        T* h_out_cub = (T*)malloc(arr_size);

        // Instantiate our radix sort algorithm with template with a typedef
        typedef Radix<T, B, E, TS> Radix4;
        int mask = GetMask(B);

        // Device allocations
        T* d_in;
        T* d_out;
        unsigned int* d_histogram1;
        unsigned int* d_histogram2;
        unsigned int* d_histogram3;
        void*         d_tmp_storage;
        cudaMalloc((void**)&d_in,  arr_size);
        cudaMalloc((void**)&d_out, arr_size);
        cudaMalloc((void**)&d_histogram1, Radix4::HistogramStorageSize(N));
        cudaMalloc((void**)&d_histogram2, Radix4::HistogramStorageSize(N));
        cudaMalloc((void**)&d_histogram3, Radix4::HistogramStorageSize(N));
        cudaMalloc((void**)&d_tmp_storage, Radix4::TempStorageSize(N, d_histogram1));

        // Allocations for cub version
        void* d_tmp_storage_cub = NULL;
        size_t tmp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_tmp_storage_cub, tmp_storage_bytes, d_in, d_out, N);
        cudaMalloc(&d_tmp_storage_cub, tmp_storage_bytes);


        // Dry runs
        Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
        RadixSortCub<T>(d_in, d_out, N, d_tmp_storage_cub, tmp_storage_bytes);
        cudaDeviceSynchronize();

        std::vector<Timer> time_our;
        std::vector<Timer> time_cub;

        for (int j = 0; j < gpu_runs; j++) {

            // Initialize the array to be sorted and transfer to device
            randomInitNat<T>(h_in, N, N);

            // Timers for our version and cub
            Timer t1, t2;

            // Move array to device
            cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

            // Run our version and save the result
            t1.Start();
            Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
            cudaDeviceSynchronize();
            t1.Stop();

            #ifdef RADIX_VALIDATE
            // Save sorted array to host for validation
            cudaMemcpy(h_out_our, d_in, arr_size, cudaMemcpyDeviceToHost);
            #endif

            // Now the CUB version
            cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);
 
            t2.Start();
            RadixSortCub<T>(d_in, d_out, N, d_tmp_storage_cub, tmp_storage_bytes);
            cudaDeviceSynchronize();
            t2.Stop();


            #ifdef RADIX_VALIDATE
            cudaMemcpy(h_out_cub, d_out, arr_size, cudaMemcpyDeviceToHost);
            // Print if we do not validate
            if (!validate<T>(h_out_our, h_out_cub, N)) {
                printf("INVALID. Size %i run %i\n", N, j);
            }
            #endif

            // Save runtimes
            time_our.push_back(t1);
            time_cub.push_back(t2);
        }

        // Save the average runtimes
        float run_our = average(time_our);
        float run_cub = average(time_cub);

        avg_our.push_back(run_our);
        avg_cub.push_back(run_cub);

        printf("Our: %.2f\n", run_our);
        printf("Cub: %.2f\n", run_cub);
        printf("factor: %f\n", run_our/run_cub);


        // Have to allocate and free each iteration of outer loop as the sizes change but they are not timed
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_histogram1);
        cudaFree(d_histogram2);
        cudaFree(d_histogram3);
        cudaFree(d_tmp_storage);
        cudaFree(d_tmp_storage_cub);
        free(h_in);
        free(h_out_our);
        free(h_out_cub);
    }

    writeRuntimes(sizes, avg_our, avg_cub, out_file);

}

template<
    typename T, 
    int B, 
    int E,
    int TS >
void benchTuning(std::vector<size_t> sizes, int gpu_runs, const char* out_file) {

    std::vector<float> avg_times;
    
    for (int i = 0; i < sizes.size(); i++) {
        size_t N = sizes[i];
        printf("===============================\n");
        printf("N: %lu\n", N);
        size_t arr_size = N * sizeof(T);


        // Host allocations
        T* h_in      = (T*)malloc(arr_size);

        // Instantiate our radix sort algorithm with template with a typedef
        typedef Radix<T, B, E, TS> Radix4;
        int mask = GetMask(B);

        // Device allocations
        T* d_in;
        T* d_out;
        unsigned int* d_histogram1;
        unsigned int* d_histogram2;
        unsigned int* d_histogram3;
        void*         d_tmp_storage;
        cudaMalloc((void**)&d_in,  arr_size);
        cudaMalloc((void**)&d_out, arr_size);
        cudaMalloc((void**)&d_histogram1, Radix4::HistogramStorageSize(N));
        cudaMalloc((void**)&d_histogram2, Radix4::HistogramStorageSize(N));
        cudaMalloc((void**)&d_histogram3, Radix4::HistogramStorageSize(N));
        cudaMalloc((void**)&d_tmp_storage, Radix4::TempStorageSize(N, d_histogram1));

        // Dry run
        Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
        cudaDeviceSynchronize();

        std::vector<Timer> times;

        for (int j = 0; j < gpu_runs; j++) {

            Timer t1;

             // Initialize the array to be sorted and transfer to device
            randomInitNat<T>(h_in, N, N);

            // Move array to device
            cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);

            // Run our version and save the result
            t1.Start();
            Radix4::Sort(d_in, d_out, N, d_histogram1, d_histogram2, d_histogram3, d_tmp_storage, mask);
            cudaDeviceSynchronize();
            t1.Stop();

            times.push_back(t1);
        }

        float avg = average(times);
        avg_times.push_back(avg);

        printf("Time: %.2f\n", avg);

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_histogram1);
        cudaFree(d_histogram2);
        cudaFree(d_histogram3);
        cudaFree(d_tmp_storage);
        free(h_in);
    }

    writeRuntimes(sizes, avg_times, avg_times, out_file);

}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Usage: ./radix <gpu runs>\n");
        return 0;
    }

    int gpu_runs = atoi(argv[1]);

    std::vector<size_t> sizes;
    sizes.push_back(100000);
    sizes.push_back(250000);
    sizes.push_back(500000);
    sizes.push_back(750000);
    sizes.push_back(1000000);
    sizes.push_back(2500000);
    sizes.push_back(5000000);
    sizes.push_back(7500000);
    sizes.push_back(10000000);
    sizes.push_back(25000000);
    sizes.push_back(50000000);
    sizes.push_back(75000000);
    sizes.push_back(100000000);
    sizes.push_back(250000000);
    sizes.push_back(500000000);
    sizes.push_back(750000000);

    // printf("===== Parameter B tests =====\n");
    // benchTuning<unsigned int, 1, 4, 256>(sizes, gpu_runs, "data/B-1-4-256.csv");
    // benchTuning<unsigned int, 4, 4, 256>(sizes, gpu_runs, "data/B-4-4-256.csv");
    // benchTuning<unsigned int, 8, 4, 256>(sizes, gpu_runs, "data/B-8-4-256.csv");

    // printf("===== Parameter E tests =====\n");
    // benchTuning<unsigned int, 4, 1, 256>(sizes, gpu_runs, "data/E-4-1-256.csv");
    benchTuning<unsigned int, 4, 4, 256>(sizes, gpu_runs, "data/E-4-4-256.csv");
    // benchTuning<unsigned int, 4, 8, 256>(sizes, gpu_runs, "data/E-4-8-256.csv");

    // printf("===== Parameter TS tests =====\n");
    // benchTuning<unsigned int, 4, 4, 256*1>(sizes, gpu_runs, "data/TS-4-4-256.csv");
    // benchTuning<unsigned int, 4, 4, 256*2>(sizes, gpu_runs, "data/TS-4-4-512.csv");
    // benchTuning<unsigned int, 4, 4, 256*3>(sizes, gpu_runs, "data/TS-4-4-768.csv");
    // benchTuning<unsigned int, 4, 4, 256*4>(sizes, gpu_runs, "data/TS-4-4-1024.csv");


    // printf("\nUnsigned int:\n");
    // bench<unsigned int, 8, 4, 512>(sizes, gpu_runs, "data/u32-8-4-512.csv");

    // printf("\nUnsigned long:\n");
    // bench<unsigned long, 8, 4, 512>(sizes, gpu_runs, "data/u64-8-4-512.csv");

    // printf("\nUnsigned short:\n");
    // bench<unsigned short, 8, 4, 512>(sizes, gpu_runs, "data/u16-8-4-512.csv");
    
    // printf("\nUnsigned char:\n");
    // bench<unsigned char, 8, 4, 512>(sizes, gpu_runs, "data/u8-8-4-512.csv");


    return 0;
}





