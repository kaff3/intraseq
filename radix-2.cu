#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "cub/device/device_scan.cuh"

#define NUM_THREADS 256
#define TILE_SIZE 1024
#define B 4
#define get_digit(V, I, M) ((V >> I) & M)
#define HISTOGRAM_SIZE 16
#define WARP 32


__global__ void kernel12(unsigned int* d_out, unsigned int* d_in, uint64_t arr_size, unsigned int* d_histogram, int curr_digit){
    int idx = threadIdx.x;

    __shared__ unsigned int s_tile[TILE_SIZE];

    // Load to shared memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int s_index = idx + i * blockDim.x;
        unsigned int d_index = blockIdx.x * blockDim.x * B + s_index;
        if (d_index < arr_size) {
            s_tile[s_index] = d_in[d_index];
        }
    }


    // Save in registers. Hopefully? Maybe?
    // unsigned int element[4];
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     // unsigned int index = idx * 4 + i;
    //     unsigned int index = idx + i * blockDim.x;
    //     if (index < arr_size) {
    //         element[i] = s_tile[index];
    //     }
    // }

    // zero initialize histogram
    __shared__ unsigned int s_histogram[HISTOGRAM_SIZE];
    if (idx < HISTOGRAM_SIZE){
        s_histogram[idx] = 0;
    }
    __syncthreads();

    // Sort in shared memory with 1-bit split. b iterations
    __shared__ unsigned int c0;
    __shared__ unsigned int c1;
    for (int i = 0; i < 4; i++) {

        // Save in registers. Hopefully? Maybe?
        unsigned int element[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int index = idx * 4 + j;
            // unsigned int index = idx + j * blockDim.x;
            if (index < arr_size) {
                element[i] = s_tile[index];
            }
        }
        
        c0 = 0;
        c1 = 0;
        __syncthreads();

        // Count
        #pragma unroll
        for (int j = 0; j < 4; j++) {    
            unsigned int s_index = idx * 4 + j;
            // unsigned int s_index = idx + j * blockDim.x;
            if (s_index < arr_size) {
                unsigned int val = element[j];
                unsigned int bit = get_digit(val, curr_digit*B + i, 0x1);
                if (bit == 0)
                    atomicAdd(&c1, 1);
            }
        } 
        __syncthreads();

        // Sort
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int s_index = idx * 4 + j;
            // unsigned int s_index = idx + j * blockDim.x;
            if (s_index < arr_size) {
                unsigned int val = element[j];
                unsigned int bit = get_digit(val, curr_digit*B + i, 0x1);
                unsigned int* adr = bit == 0 ? &c0 : &c1; 
                unsigned int old = atomicAdd(adr, 1);
                s_tile[old] = val;
            }
        } 
    }
    __syncthreads();

    // // Print s_tile
    // for (int i = 0; i < 4; i++) {
    //     unsigned int index = idx + i * blockDim.x;
    //     if (index < arr_size) {
    //         printf("s_tile[%3i] = %u\n", index, s_tile[index]);
    //     }
    // } __syncthreads();

    // Compute final histogram
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // unsigned int s_index = idx + i * blockDim.x;
        unsigned int s_index = idx * 4 + i;
        if (s_index < arr_size) {
            unsigned int val   = s_tile[s_index];
            unsigned int digit = get_digit(val, curr_digit*4, 0xF);
            atomicAdd(s_histogram + digit, 1);
        }
    }
    __syncthreads();

    // Write histogram to global memory
    if (idx < HISTOGRAM_SIZE) {
        d_histogram[gridDim.x * idx + blockIdx.x] = s_histogram[idx];
    }
    __syncthreads();

    // Write sorted tile to global memory
    for (int i = 0; i < 4; i++) {
        unsigned int s_index = idx + i * blockDim.x;
        unsigned int d_index = blockIdx.x * blockDim.x * B + s_index;
        if (d_index < arr_size) {
            d_out[d_index] = s_tile[s_index];
        }
    }
    // __syncthreads();
}



// form sorting_test.cu
void randomInitNat(unsigned int* data, const unsigned int size, const unsigned int H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

int main(int argc, char* argv[]){

    const uint64_t N = atoi(argv[1]);
    // TODO: maybe check N if it is too big
    const uint64_t arr_size = N * sizeof(unsigned int);

    // Host allocations
    unsigned int* h_in  = (unsigned int*) malloc(arr_size);
    unsigned int* h_out = (unsigned int*) malloc(arr_size);

    // Create random array to sort
    randomInitNat(h_in, N, 0xF);

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


    unsigned int* d_res;
    for (int i = 0; i < (sizeof(unsigned int)*8)/B; i++) {
        printf("=============================================\n");
        kernel12<<< num_blocks, NUM_THREADS >>>(d_out, d_in, N, d_histogram, i);

        // Swap input input and output
        // unsigned int* tmp = d_out;
        
        // d_res = d_out;
        // d_out = d_in;
        // d_in  = d_res;

        d_res = d_in;
        d_in  = d_out;
        d_out = d_res;

        cudaMemcpy(h_out, d_res, arr_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            printf("%10x      %10x\n", h_out[i], h_in[i]);
        }
    }

    printf("d_out: %x\n", d_out);
    printf("d_in : %x\n", d_in);
    printf("d_res: %x\n", d_res);

    cudaMemcpy(h_out, d_res, arr_size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     printf("%10x      %10x\n", h_out[i], h_in[i]);
    // }



    // kernel12



    // kernel 3
    // void     *d_temp_storage = NULL;
    // size_t   temp_storage_bytes = 0;
    // cub::DeviceScan::ExclusiveSum();
    
    
    // kernel 4


    // Clean up memory

    return 0;
}
