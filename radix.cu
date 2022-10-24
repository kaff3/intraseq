#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "cub/device/device_scan.cuh"

#define NUM_THREADS 256
#define TILE_SIZE 1024
#define B 4
#define get_digit(V, I) ((V >> (I * 4)) & 0xF)
#define HISTOGRAM_SIZE 16
#define WARP 32


__global__ void kernel12(unsigned int* d_out, unsigned int* d_in, uint64_t arr_size, unsigned int* d_histogram, int curr_digit){
    __shared__ unsigned int s_tile[TILE_SIZE];
    __shared__ unsigned int s_tile_sorted[TILE_SIZE];
    __shared__ unsigned int s_histogram[HISTOGRAM_SIZE];
    int iterations = TILE_SIZE / NUM_THREADS; 
    int idx = threadIdx.x;
    // int gid = blockDim.x * blockIdx.x + threadIdx.x; 

    // zero initialize histogram
    if (idx < HISTOGRAM_SIZE){
        s_histogram[idx] = 0;
    }
    __syncthreads();



    // Copy elements from global memory and increment histogram
    for (int i = 0; i < iterations; i++) {
        // uint64_t arr_i = gid + i * NUM_THREADS;
        uint64_t arr_i = blockIdx.x * blockDim.x * B + (idx + i * NUM_THREADS);
        if (arr_i < arr_size) {
            unsigned int val = d_in[arr_i];
            unsigned int digit = get_digit(val, curr_digit);
            atomicAdd(s_histogram + digit, 1);
            
            // copy to shared memory
            s_tile[idx + i * NUM_THREADS] = val;
        }
    } __syncthreads();
    
    // copy local histogram to global memory transposed
    // Because there are 8 warps per block, if for each Warp
    // the two first threads are used, there is better
    // coalesed access.
    // const unsigned int lane = idx & (WARP-1);
    // if (lane == 0 || lane == 1){
    //     unsigned int curr_warp = idx / WARP;
    //     int histogram_i = curr_warp * 2 + lane;
    //     // p = num blocks = gridDim.x
    //     d_histogram[gridDim.x * histogram_i + blockIdx.x] = s_histogram[histogram_i];
    // }

    if (idx < 16) {
        d_histogram[gridDim.x * idx + blockIdx.x] = s_histogram[idx];
    } __syncthreads();

    // if (idx == 0) {
    //     printf("s_histogram original\n");
    //     for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    //         printf("his[%3i] = %3i\n", i, s_histogram[i]);
    //     }
    // } __syncthreads();

    // exclusive scan over histogram
    // TODO: currently sequential on a single thread
    if (idx == 0){
        for (int i = HISTOGRAM_SIZE-1; i > 0; i--){
            s_histogram[i] = s_histogram[i-1];
        }
        s_histogram[0] = 0;
        for (int i = 1; i < HISTOGRAM_SIZE; i++){
            s_histogram[i] += s_histogram[i-1];
        }
    } __syncthreads();


    // if (idx == 0) {
    //     printf("s_histogram scanned\n");
    //     for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    //         printf("his[%3i] = %3i\n", i, s_histogram[i]);
    //     }
    // } __syncthreads();

    for (int i  = 0; i < iterations; i++){
        // foreach val in s_tile

        // unsigned int val = s_tile[idx + i * NUM_THREADS];
        // unsigned int old = atomicAdd(s_histogram + val, 1);

        unsigned int index = idx + i * NUM_THREADS;
        if (index < arr_size) {
            unsigned int val   = s_tile[index];
            unsigned int digit = get_digit(val, curr_digit);
            unsigned int old   = atomicAdd(s_histogram + digit, 1);
            s_tile_sorted[old] = val;
        }
    } __syncthreads();


    // SKAL VI SKRIVE DET SORTEREDE TILBAGE?
    // KAN MAN IKKE BARE LADE DET LIGGE I SHARED OG VENTE TIL STEP 4?
    for (int i = 0; i < iterations; i++){
        // d_out[gid + i * NUM_THREADS] = s_tile_sorted[idx + i * NUM_THREADS];
        
        unsigned int index  = idx + i * NUM_THREADS;
        unsigned int offset = blockIdx.x * blockDim.x * B;
        if (offset + index < arr_size) {
            d_out[offset + index] = s_tile_sorted[index];
        }
    } __syncthreads();
}

// 
// __global__ void kernel3(unsigned int* d_out, unsigned int* d_in, unsigned int* d_histogram){
    
//     cub::DeviceScan::ExclusiveSum()

// }

__global__ void kernel4(unsigned int* d_out, unsigned int* d_in, unsigned int* d_histogram){

}

// form sorting_test.cu
void randomInitNat(unsigned int* data, const unsigned int size, const unsigned int H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

int main(int argc, char* argv[]){


    // unsigned int* vals = (unsigned int*) malloc(10000 * sizeof(unsigned int));

    const uint64_t N = atoi(argv[1]);
    // TODO: maybe check N if it is too big
    const uint64_t arr_size = N * sizeof(unsigned int);

    // Host allocations
    unsigned int* h_in  = (unsigned int*) malloc(arr_size);
    unsigned int* h_out = (unsigned int*) malloc(arr_size);

    // Create random array to sort
    randomInitNat(h_in, N, 0xFF);

    // Compute blocks and block sizes
    unsigned int num_blocks = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Device allocations
    unsigned int* d_in;
    unsigned int* d_out;
    unsigned int* d_histogram;
    cudaMalloc((void**)&d_in,  arr_size);
    cudaMalloc((void**)&d_out, arr_size);
    cudaMalloc((void**)&d_histogram, num_blocks * HISTOGRAM_SIZE * sizeof(unsigned int));

    // Copy initial array to device
    cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);


    printf("num blocks: %i\n", num_blocks);
    printf("num threads: %i\n", num_blocks*NUM_THREADS);

    printf("d_out: %x\n", d_out);
    printf("d_in : %x\n", d_in);

    unsigned int* d_res;
    for (int i = 0; i < (sizeof(unsigned int)*8)/B; i++) {

        kernel12<<< num_blocks, NUM_THREADS >>>(d_out, d_in, N, d_histogram, i);

        // Swap input input and output
        // unsigned int* tmp = d_out;
        
        // d_res = d_out;
        // d_out = d_in;
        // d_in  = d_res;

        d_res = d_in;
        d_in  = d_out;
        d_out = d_res;

        // cudaMemset(d_histogram, 0, num_blocks*HISTOGRAM_SIZE*sizeof(int));
    }
    // kernel12<<< num_blocks, NUM_THREADS >>>(d_out, d_in, N, d_histogram, 0);

    printf("d_res: %x\n", d_res);

    cudaMemcpy(h_out, d_res, arr_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%10i      %10i\n", h_out[i], h_in[i]);
    }



    // kernel12



    // kernel 3
    // void     *d_temp_storage = NULL;
    // size_t   temp_storage_bytes = 0;
    // cub::DeviceScan::ExclusiveSum();
    
    
    // kernel 4


    // Clean up memory

    return 0;
}
