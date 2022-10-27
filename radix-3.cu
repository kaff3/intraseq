#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "cub/device/device_scan.cuh"


#define THREAD_ELEMENTS     4   // 
#define NUM_THREADS         256
#define TILE_SIZE           (NUM_THREADS * THREAD_ELEMENTS)
#define B                   4
#define HISTOGRAM_SIZE      16  // 2^B
#define GET_DIGIT(V, I, M)  ((V >> I) & M)

// #define NUM_POSITIONS       (NUM_THREADS * 2)


__global__ void kernel12(unsigned int* d_out, unsigned int* d_in, uint64_t arr_size, unsigned int* d_histogram, int curr_digit){
    // int idx = threadIdx.x;

    __shared__ unsigned int s_tile[TILE_SIZE];

    // Load to shared memory
    #pragma unroll
    for (int i = 0; i < THREAD_ELEMENTS; i++) {
        unsigned int s_index = threadIdx.x + i * blockDim.x;
        unsigned int d_index = blockIdx.x * TILE_SIZE + s_index;
        if (d_index < arr_size) {
            s_tile[s_index] = d_in[d_index];
        }
    }
    __syncthreads();


    // Sort in shared memory b iterations 1-bit split
    unsigned int elements[THREAD_ELEMENTS];
    for (int i = 0; i < B; i++) {

        // Read elements
        unsigned int index;
        #pragma unroll
        for (int j = 0; j < THREAD_ELEMENTS; j++) {
            index = threadIdx.x * THREAD_ELEMENTS + j;
            if (index < arr_size) {
                elements[j] = s_tile[index];
            }
        }
        __syncthreads();

        unsigned int ps0 = 0;
        unsigned int ps1 = 0;

        #pragma unroll
        for (int j = 0; j < THREAD_ELEMENTS; j++) {
            unsigned int index = threadIdx.x * THREAD_ELEMENTS + j;
            if (index < arr_size) {
                unsigned int bit = GET_DIGIT(elements[j], curr_digit*B+i, 0x1);
                ps0 += (bit == 0 ? 1 : 0);
                ps1 += (bit == 1 ? 1 : 0);
            }
        }
        __syncthreads();


        static const int num_positions = 2 * NUM_THREADS;
        __shared__ unsigned int positions[num_positions];

        // Write positions to shared memory to prepare for scan
        positions[threadIdx.x]              = ps0;
        positions[blockDim.x + threadIdx.x] = ps1;
        __syncthreads();

        // Perform scan. TODO: Make it not sequential
        if (threadIdx.x == 0) {
            for (int j = 1; j < num_positions; j++) {
                positions[j] += positions[j-1];
            }
        }
        __syncthreads();

        ps0 = (threadIdx.x == 0 ? 0 : positions[threadIdx.x - 1]);
        ps1 = positions[blockDim.x + threadIdx.x - 1];

        // Sort by scattering
        #pragma unroll
        for (int j = 0; j < THREAD_ELEMENTS; j++) {
            unsigned int index = threadIdx.x * THREAD_ELEMENTS + j;
            if (index < arr_size) {
                unsigned int bit = GET_DIGIT(elements[j], curr_digit*B+i, 0x1);
                unsigned int pos = (bit == 0 ? ps0 : ps1);
                ps0 += (bit == 0 ? 1 : 0);
                ps1 += (bit == 1 ? 1 : 0);
                s_tile[pos] = elements[j];
            }
        }
        __syncthreads(); // For next iteration

    } // Big loop end



    // zero initialize histogram
    __shared__ unsigned int s_histogram[HISTOGRAM_SIZE];
    if (threadIdx.x < HISTOGRAM_SIZE){
        s_histogram[threadIdx.x] = 0;
    }
    __syncthreads();

        
    // Compute final histogram
    for (int i = 0; i < TILE_SIZE; i++) {
        // TODO: Fix so the mask 0xF is based of B, ie the number of bits in digit
        unsigned int digit = GET_DIGIT(s_tile[i], curr_digit*B, 0xF);
        atomicAdd(s_histogram + digit, 1);
    }

    // Write histogram to global memory
    if (threadIdx.x < HISTOGRAM_SIZE) {
        d_histogram[gridDim.x * threadIdx.x + blockIdx.x] = s_histogram[threadIdx.x];
    }
    __syncthreads();

    // Write sorted til back to global memory. coalesced
    for (int i = 0; i < THREAD_ELEMENTS; i++) {
        unsigned int s_index = threadIdx.x + blockDim.x * i;
        unsigned int d_index = blockIdx.x * TILE_SIZE + s_index;
        if (d_index < arr_size) {
            d_out[d_index] = s_tile[s_index];
        }
    }
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
    randomInitNat(h_in, N, N);

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


    printf("num blocks:  %i\n", num_blocks);
    printf("num threads: %i\n", num_blocks*NUM_THREADS);
    printf("tile size:   %i\n", TILE_SIZE);


    for (int i = 0; i < (sizeof(unsigned int)*8)/B; i++) {
        kernel12<<< num_blocks, NUM_THREADS >>>(d_out, d_in, N, d_histogram, i);

        // Swap input input and output
        // unsigned int* tmp = d_out;
        
        // d_res = d_out;
        // d_out = d_in;
        // d_in  = d_res;

        unsigned int* tmp;
        tmp = d_in;
        d_in  = d_out;
        d_out = tmp;
    }

    // kernel12<<< num_blocks, NUM_THREADS >>>(d_out, d_in, N, d_histogram, 0);

    // Copy from device to print result
    cudaMemcpy(h_out, d_out, arr_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%10x      %10x\n", h_out[i], h_in[i]);
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
