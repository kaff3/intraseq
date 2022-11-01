#ifndef PMPH_RADIX
#define PMPH_RADIX

#include<cuda_runtime.h>
#include <cooperative_groups.h>
#include"cub/cub.cuh"

#include<stdio.h>
#include<stdint.h>

#define GET_DIGIT(V, I, M)  ((V >> I) & M)

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

    // B itersions of 1 bit splits sorting locally in s_tile
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
                ps0 += (bit == 0 ? 1 : 0);
                ps1 += (bit == 1 ? 1 : 0);
            }
        }
        __syncthreads();

        // Scan
        typedef cub::BlockScan<T, TS> BlockScan;
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
                ps0 += (bit == 0 ? 1 : 0);
                ps1 += (bit == 1 ? 1 : 0);
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
            unsigned int elmDigit = GET_DIGIT(elements[i], digit*B, mask);
            atomicAdd(s_histogram + elmDigit, 1);
        }
    }
    __syncthreads();


    // Write histogram to global memory transposed
    if (threadIdx.x < HISTOGRAM_ELEMENTS) {
        d_histogram[gridDim.x * threadIdx.x + blockIdx.x] = s_histogram[threadIdx.x];
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
globalScatterKernel(T* d_in, T* d_out, int N, unsigned int* d_histogram, int digit, int mask) {

    int tile_id = blockIdx.x;
    int last_tile = ((N + TILE_ELEMENTS - 1) / TILE_ELEMENTS) - 1;
    int last_size = N % TILE_ELEMENTS == 0 ? TILE_ELEMENTS : N % TILE_ELEMENTS;
    int local_tile_size = tile_id == last_tile ? last_size : TILE_ELEMENTS;

    __shared__ T s_tile[TILE_ELEMENTS];
    loadTile<T, E, TILE_ELEMENTS>(s_tile, d_in, N);
    __syncthreads();

    T elements[E];
    loadThreadElements<T, E>(elements, s_tile, local_tile_size);
    __syncthreads();

    __shared__ unsigned int s_histogram[HISTOGRAM_ELEMENTS];
    if (threadIdx.x < HISTOGRAM_ELEMENTS) {
        s_histogram[threadIdx.x] = d_histogram[gridDim.x * threadIdx.x + blockIdx.x];
    }
    __syncthreads();

    unsigned int ps[HISTOGRAM_ELEMENTS] = {0};
    #pragma unroll
    for (int i = 0; i < E; i++) {
        unsigned int index = threadIdx.x * E + i;
        if (index < local_tile_size){
            unsigned int val = GET_DIGIT(elements[i], digit*B, mask);
            ps[val] += 1;
        }
    }
    __syncthreads();

    typedef cub::BlockScan<T, TS> BlockScan;
    for (int i = 0; i < HISTOGRAM_ELEMENTS; i++) {
        __shared__ typename BlockScan::TempStorage tmp;
        unsigned int offset = s_histogram[i];
        BlockScan(tmp).ExclusiveScan(ps[i], ps[i], offset, cub::Sum());
        __syncthreads();
    } 
    __syncthreads();

    // Scatter to global memory
    #pragma unroll
    for (int i = 0; i < E; i++) {
        unsigned int s_index = threadIdx.x * E + i;
        unsigned int d_index = blockIdx.x * TILE_ELEMENTS + s_index;
        if (d_index < N) {
            unsigned int val = GET_DIGIT(elements[i], digit*B, mask);
            unsigned int old = ps[val];
            ps[val] += 1;
            d_out[old] = elements[i];
        }
    }

} // End globalScatterKernel




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

    static void globalRanking(int N, unsigned int* d_histogram, unsigned int* d_histogram_scanned) {
        int num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
        void* d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveScan(d_tmp_storage, tmp_storage_bytes, d_histogram, d_histogram_scanned, cub::Sum(), 0, HISTOGRAM_ELEMENTS*num_blocks);
        cudaMalloc(&d_tmp_storage, tmp_storage_bytes);
        cub::DeviceScan::ExclusiveScan(d_tmp_storage, tmp_storage_bytes, d_histogram, d_histogram_scanned, cub::Sum(), 0, HISTOGRAM_ELEMENTS*num_blocks);
        cudaFree(d_tmp_storage);
    }

public:
        
    // Constructor and Destructor empty on purpose
    Radix() {}
    ~Radix() {}

    // Main function of the class. Sorts the data
    static void Sort(T* d_in, T* d_out, size_t N, unsigned int* d_histogram, int mask) {

        int num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;

        unsigned int* d_histogram_scanned;
        cudaMalloc((void**)&d_histogram_scanned, HISTOGRAM_SIZE*num_blocks);

        // unsigned int* h_histogram = (unsigned int*)malloc(HISTOGRAM_SIZE*num_blocks);
        // unsigned int* h_histogram_scanned = (unsigned int*)malloc(HISTOGRAM_SIZE*num_blocks);

        // Print some debug informtion
        printf("num blcoks: %i\n", num_blocks);
        printf("tile size:  %i\n", TILE_ELEMENTS);


        int iterations = sizeof(T)*8 / B;
        for (int i = 0; i < iterations; i++) {
            // Kernel 1 + 2
            rankKernel<T, B, E, TS, TILE_ELEMENTS, HISTOGRAM_ELEMENTS>
                <<<num_blocks, TS>>>(d_in, d_out, N, d_histogram, i, mask);

            // Kernel 3
            globalRanking(N, d_histogram, d_histogram_scanned);

            // Kernel 4
            globalScatterKernel<T, B, E, TS, TILE_ELEMENTS, HISTOGRAM_ELEMENTS>
                <<<num_blocks, TS>>>(d_out, d_in, N, d_histogram_scanned, i, mask);

        }
        cudaDeviceSynchronize();

        cudaFree(d_histogram_scanned);

    }

    static size_t d_histogramSize(int N) {
        int num_blocks = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
        return num_blocks * HISTOGRAM_SIZE;
    }

}; // Radix end



#endif