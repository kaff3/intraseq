#ifndef PMPH_RADIX
#define PMPH_RADIX

#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include"cub/cub.cuh"
#include"transpose-kernels.cu.h"
#include"transpose-host.cu.h"
#include"timing.h"

#include<stdio.h>
#include<stdint.h>

#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "./helper.cu.h"

#define GET_DIGIT(V, I, M)  ((V >> I) & M)
#define TRANSPOSE_T 32
/******************************************************************************
 * Device and Global fuctions
******************************************************************************/

template<
    typename T,     // The type of the data to be sorted
    int E,          // The number of elements pr. thread
    int TILE_ELEMENTS   
>
__device__ void loadTile(T* tile, T* d_in, size_t N) {
        #pragma unroll
        for (int i = 0; i < E; i++) {
            size_t s_index = threadIdx.x + i * blockDim.x;
            size_t d_index = blockIdx.x * TILE_ELEMENTS + s_index;
            if (d_index < N) {
                tile[s_index] = d_in[d_index];
            }
        }
    }


template<
    typename T,     // The type of the data to be sorted
    int E           // The number of elements pr. thread
>
__device__ void loadThreadElements(T* elements, T* tile, size_t N) {
    #pragma unroll
    for (int i = 0; i < E; i++) {
        size_t index = threadIdx.x * E + i;
        if (index < N) {
            elements[i] = tile[index];
        }
    }
}


template<
    typename T,     // The type of the data to be sorted
    int B,          // The amount of bits that make up a digit
    int E,          // The number of elements pr. thread
    int TS,         // The number of threads pr. block
    int TILE_ELEMENTS,
    int HISTOGRAM_ELEMENTS >
__global__ void 
rankKernel(T* d_in, T* d_out, size_t N, unsigned int* d_histogram, int digit, int mask) {
    
    int tile_id = blockIdx.x;
    int last_tile = ((N + TILE_ELEMENTS - 1) / TILE_ELEMENTS) - 1;
    int last_size = N % TILE_ELEMENTS == 0 ? TILE_ELEMENTS : N % TILE_ELEMENTS;
    int local_tile_size = tile_id == last_tile ? last_size : TILE_ELEMENTS;

    __shared__ T s_tile[TILE_ELEMENTS];
    loadTile<T, E, TILE_ELEMENTS>(s_tile, d_in, N);
    __syncthreads();

    T elements[E];

    // B iterations of 1 bit splits sorting locally in s_tile
    for (int i = 0; i < B; i++) {
        loadThreadElements<T, E>(elements, s_tile, local_tile_size);

        // Count
        unsigned int ps0 = 0;
        unsigned int ps1 = 0; 
        #pragma unroll
        for (int j = 0; j < E; j++) {
            size_t index = threadIdx.x * E + j;
            if (index < local_tile_size) {
                T val = elements[j];
                T bit = ((val >> (digit*B+i)) & 0x1);
                ps0 += bit ^ 0x1;
                ps1 += bit;
            }
        }
        __syncthreads();

        // Scan
        typedef cub::BlockScan<unsigned int, TS> BlockScan;
        __shared__ union {
            typename BlockScan::TempStorage ps0;
            typename BlockScan::TempStorage ps1;
        } ps_storage;
        unsigned int aggregate;

        BlockScan(ps_storage.ps0).ExclusiveScan(ps0, ps0, 0, cub::Sum(), aggregate);
        __syncthreads();
        BlockScan(ps_storage.ps1).ExclusiveScan(ps1, ps1, aggregate, cub::Sum());
        __syncthreads();

        // Scatter
        #pragma unroll
        for (int j = 0; j < E; j++) {
            size_t index = threadIdx.x * E + j;
            if (index < local_tile_size) {
                T val = elements[j];
                T bit = ((val >> (digit*B+i)) & 0x1);
                unsigned int old = (bit == 0 ? ps0 : ps1);
                ps0 += bit ^ 0x1;
                ps1 += bit;
                s_tile[old] = val;
            }
        }
        __syncthreads(); // Sync for next iteration
    }

    // Compute final histogram
    __shared__ unsigned int s_histogram[HISTOGRAM_ELEMENTS];
    if (threadIdx.x < HISTOGRAM_ELEMENTS){
        s_histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < E; i++) {
        size_t index = threadIdx.x * E + i;
        if (index < local_tile_size) {
            T elmDigit = GET_DIGIT(elements[i], digit*B, mask);
            atomicAdd(s_histogram + elmDigit, 1);
        }
    }
    __syncthreads();


    // Write histogram to global memory coalesced
    if (threadIdx.x < HISTOGRAM_ELEMENTS) {
        d_histogram[HISTOGRAM_ELEMENTS * blockIdx.x + threadIdx.x] = s_histogram[threadIdx.x];
    }


    // Write sorted tile back to global memory. coalesced
    #pragma unroll
    for (int i = 0; i < E; i++) {
        unsigned int s_index = threadIdx.x + blockDim.x * i;
        unsigned int d_index = blockIdx.x * TILE_ELEMENTS + s_index;
        if (d_index < N) {
            d_out[d_index] = s_tile[s_index];
        }
    }
} // end rankKernel


template <
    typename T,     // The type of the data to be sorted
    int B,          // The amount of bits that make up a digit
    int E,          // The number of elements pr. thread
    int TS,         // The number of threads pr. block
    int TILE_ELEMENTS,
    int HISTOGRAM_ELEMENTS >
__global__ void 
globalScatterKernel(T* d_in, T* d_out, size_t N, unsigned int* d_histogram, unsigned int* d_histogram_scan, 
                    int digit, int mask) {
    int tid = threadIdx.x;

    // load histograms into shared memory
    __shared__ unsigned int s_histogram[HISTOGRAM_ELEMENTS];
    __shared__ unsigned int s_histogram_global_scan[HISTOGRAM_ELEMENTS];
    __shared__ unsigned int s_histogram_local_scan[HISTOGRAM_ELEMENTS];

    if (tid < HISTOGRAM_ELEMENTS) {
        s_histogram[tid] = d_histogram[HISTOGRAM_ELEMENTS * blockIdx.x + tid];
    }

    if (tid < HISTOGRAM_ELEMENTS) {
        s_histogram_global_scan[tid] = d_histogram_scan[HISTOGRAM_ELEMENTS * blockIdx.x + tid];
    }
    __syncthreads();

    // Scan across threads in block to create the locally scanned histogram.
    typedef cub::BlockScan<unsigned int, TS> BlockScan;
    __shared__ typename BlockScan::TempStorage count;
    unsigned int in = s_histogram[tid % HISTOGRAM_ELEMENTS];
    unsigned int out = 0;
    BlockScan(count).ExclusiveScan(in, out, 0, cub::Sum());
    if (tid < HISTOGRAM_ELEMENTS) {
        s_histogram_local_scan[tid] = out;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < E; i++){
        size_t loc_idx = tid + (i * TS);
        size_t index = blockIdx.x * TILE_ELEMENTS + loc_idx;

        if (index < N){
            T full_val = d_in[index];
            T val = GET_DIGIT(full_val, digit*B, mask);
            
            // global_pos: position in global array where this block should place its values with the found digits
            // local_pos:  position relative to other values with same digit in this block
            unsigned int global_pos = s_histogram_global_scan[val];
            unsigned int local_pos = (s_histogram[val] == 1) ? 0 : loc_idx - s_histogram_local_scan[val];  
            unsigned int pos = global_pos + local_pos;

            // scatter
            d_out[pos] = full_val;            
        }
    }
}



/******************************************************************************
 * Structs
******************************************************************************/

/*
 * This struct should be instantiated with the corect template parameters after
 * which the static Sort method can be called, which will itself instatiate kernels
 * with the correct "constants", also as template parameters. It was done this way as
 * __global__ functions are not able to be defined as a member of either structs or
 * classes.
*/

template<
    typename T,     // The type of the data to be sorted
    int B,          // The amount of bits that make up a digit
    int E,          // The number of elements pr. thread
    int TS          // The number of threads pr. block
>
struct Radix {
private:
    static const size_t HISTOGRAM_ELEMENTS = 1 << B;
    static const size_t HISTOGRAM_SIZE     = sizeof(unsigned int) * HISTOGRAM_ELEMENTS;
    static const size_t TILE_ELEMENTS      = TS * E;

public:
        
    // Constructor and Destructor empty on purpose
    // Radix() {}
    // ~Radix() {}

    static size_t HistogramStorageSize(size_t N) {
        size_t num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
        return num_blocks * HISTOGRAM_SIZE;
    }

    static size_t TempStorageSize(size_t N, unsigned int* d_histogram) {
        size_t num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
        size_t size = 0;
        cub::DeviceScan::ExclusiveScan(NULL, size, d_histogram, d_histogram, cub::Sum(), 0, HISTOGRAM_ELEMENTS*num_blocks);
        return size;
    } 

    // Main function of the class. Sorts the data
    static void Sort(T* d_in, T* d_out, size_t N, 
        unsigned int* d_histogram, unsigned int* d_histogram_scan, unsigned int* d_histogram_transpose,
        void* d_tmp_storage, int mask) {

        size_t num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;

        size_t tmp_storage_bytes = TempStorageSize(N, d_histogram);

        int iterations = sizeof(T)*8 / B;
        for (int i = 0; i < iterations; i++) {

            rankKernel<T, B, E, TS, TILE_ELEMENTS, HISTOGRAM_ELEMENTS>
                <<<num_blocks, TS>>>(d_in, d_out, N, d_histogram, i, mask);

            // transpose
            transposeTiled<unsigned int, 32>(d_histogram, d_histogram_transpose, num_blocks, HISTOGRAM_ELEMENTS);
            // scan
            cub::DeviceScan::ExclusiveScan(d_tmp_storage, tmp_storage_bytes, d_histogram_transpose, d_histogram_scan, cub::Sum(), 0, (int)HISTOGRAM_ELEMENTS*num_blocks);
            // transpose
            transposeTiled<unsigned int, 32>(d_histogram_scan, d_histogram_transpose, HISTOGRAM_ELEMENTS, num_blocks);
            
            unsigned int* tmp;
            tmp = d_histogram_scan;
            d_histogram_scan = d_histogram_transpose;
            d_histogram_transpose = tmp;

            globalScatterKernel<T, B, E, TS, TILE_ELEMENTS, HISTOGRAM_ELEMENTS>
                <<<num_blocks, TS>>>(d_out, d_in, N, d_histogram, d_histogram_scan, i, mask);
        }
    }
}; // Radix end



#endif
