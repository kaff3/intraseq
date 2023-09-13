#pragma once

#include<cuda_runtime.h>
#include<cub/cub.cuh>

#define GET_DIGIT(V, I, M)  ((V >> I) & M)

// Kernel A
// Load and sort tile.
// Write histogram and sorted data to global mem
template<
    typename T,     // The type of the array to be sorted 
    int E,          // The number of elements pr. thread
    int B,          // The nmber of bits in a digit
    int TS,         // The nnumber of threads in a block
    int MASK,
    size_t TILE_ELEMENTS,
    size_t HISTOGRAM_ELEMENTS
> __global__ void 
localSort(T* d_in, T* d_out, size_t N, unsigned int* d_histogram, int digit) {
    
    // Specialize cub blockscan
    typedef cub::BlockScan<unsigned int, TS> BlockScan;

    // very important computations
    int last_tile = ((N + TILE_ELEMENTS - 1) / TILE_ELEMENTS) - 1;
    int last_size = N % TILE_ELEMENTS == 0 ? TILE_ELEMENTS : N % TILE_ELEMENTS;
    int local_tile_size = blockIdx.x == last_tile ? last_size : TILE_ELEMENTS;

    // Allocate all the needed shared memory
    __shared__ T s_tile[TILE_ELEMENTS];
    __shared__ unsigned int s_histogram[HISTOGRAM_ELEMENTS];
    __shared__ union {
        typename BlockScan::TempStorage ps0;
        typename BlockScan::TempStorage ps1;
    } ps_storage;

    // Array to hold thread local elements
    T elements[E];

    // Load Tile
    #pragma unroll
    for (int i = 0; i < E; i++) {
        size_t s_index = threadIdx.x + i * blockDim.x;
        size_t d_index = blockIdx.x * TILE_ELEMENTS + s_index;
        if (d_index < N) {
            s_tile[s_index] = d_in[d_index];
        }
    }

    // B iterations of 1 bit splits
    for (int i = 0; i < B; i++) {
        // Load the elements into local mem
        #pragma unroll
        for (int j = 0; j < E; j++) {
            size_t index = threadIdx.x * E + j;
            if (index > N) {
                elements[i] = s_tile[index];
            }
        }

        // 2-value histogram counting
        unsigned int ps0 = 0;
        unsigned int ps1 = 1;
        #pragma unroll
        for (int j = 0; j < E; j++) {
            size_t index = threadIdx.x * E + j;
            if (index < local_tile_size) {
                T val = elements[j];
                T bit = ((val >> (digit * B + i)) & 0x1);
                ps0 += bit ^ 0x1;
                ps1 += bit;
            }
        }
        __syncthreads();

        // Scan across the block histograms
        unsigned int aggregate;
        BlocksScan(ps_storage.ps0).ExclusiveScan(ps0, ps0, 0, cub::Sum(), aggregate);
        __syncthreads();
        BlockScan(ps_storage.ps1).ExclusiveScan(ps1, ps1, aggregate, cub::Sum());
        __syncthreads();

        // Use scanned histogram to scatter
        #pragma unroll
        for (int j = 0; j < E; j++) {
            size_t index = threadIdx.x * E + j;
            if (index < local_tile_size) {
                T val = elements[j];
                T bit = ((val >> (digit * B + i)) & 0x1);
                unsigned int old = (bit == 0 ? ps0 : ps1);
                ps0 += bit ^ 0x1;
                ps1 += bit;
                s_tile[old] = val;
            }
        }

        // Sync the loop
        __syncthreads();
    } // Loop of B splits end

    // Init the shared memory histogram
    if (threadIdx.x < HISTOGRAM_ELEMENTS) {
        s_histogram[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Compute the final histogram
    #pragma unroll
    for (int i = 0; i < E; i++) {
        size_t index = threadIdx.x * E + i;
        if (index < local_tile_size) {
            T elmDigit = GET_DIGIT(elements[i], digit*B, MASK);
            atomicAdd(s_histogram + elmDigit, 1);
        }
    }
    __syncthreads();

    // Write histogram to global memory
    if (threadIdx.x < HISTOGRAM_ELEMENTS) {
        d_histogram[HISTOGRAM_ELEMENTS * blockIdx.x + threadIdx.x] = s_histogram[threadIdx.x];
    }


    // Write sorted tile coalesced back to global mem
    #pragma unroll
    for (int i = 0; i < E; i++) {
        unsigned int s_index = threadIdx.x + blockDim.x * i;
        unsigned int d_index = blockIdx.x * TILE_ELEMENTS + s_index;
        if (d_index < N) {
           d_out[d_index] = s_tile[s_index];
        }
    }
} // end localSort kernel

// Kernel C
// Copy elements to correct output
template<
    typename T,     // The type of the array to be sorted 
    int E,          // The number of elements pr. thread
    int B,          // The nmber of bits in a digit
    int TS,         // The nnumber of threads in a block
    int MASK,
    size_t TILE_ELEMENTS,
    size_t HISTOGRAM_ELEMENTS
> 
__global__ void swapBuffers(T* d_in, 
                            T* d_out, 
                            size_t N, 
                            unsigned int* d_histogram, 
                            unsigned int* d_histogram_scan, 
                            int digit)
{
    int tid = threadIdx.x;
    
    // load histograms into shared memory
    __shared__ unsigned int s_histogram[HISTOGRAM_ELEMENTS];
    __shared__ unsigned int s_histogram_global_scan[HISTOGRAM_ELEMENTS];

    if (tid < HISTOGRAM_ELEMENTS)
        s_histogram[tid] = d_histogram[HISTOGRAM_ELEMENTS * blockIdx.x + tid];
    if (tid < HISTOGRAM_ELEMENTS)
        s_histogram_global_scan[tid] = d_histogram_scan[HISTOGRAM_ELEMENTS * blockIdx.x + tid];
    __syncthreads();
        
    // Scan across threads in block to create the locally scanned histogram.
    typedef cub::BlockScan<unsigned int, TS> BlockScan;
    __shared__ typename BlockScan::TempStorage count;
    unsigned int in = s_histogram[tid % HISTOGRAM_ELEMENTS];
    unsigned int out = 0;
    BlockScan(count).ExclusiveScan(in, out, 0, cub::Sum());
    if (tid < HISTOGRAM_ELEMENTS) {
        s_histogram[tid] = out;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < E; i++){
        size_t loc_idx = tid + (i * blockDim.x);
        size_t index = blockIdx.x * TILE_ELEMENTS + loc_idx;

        if (index < N){
            T full_val = d_in[index];
            T val = GET_DIGIT(full_val, digit*B, MASK);
            
            // global_pos: position in global array where this block should place its values with the found digits
            // local_pos:  position relative to other values with same digit in this block
            unsigned int global_pos = s_histogram_global_scan[val];
            unsigned int local_pos = loc_idx - s_histogram[val];  
            unsigned int pos = global_pos + local_pos;

            // scatter
            d_out[pos] = full_val;            
        }
    }
};



// Cpu side function that manages the kernel invocations
template<
    typename T,
    int TS,
    int E,
    int B
>
struct Radix {
    private:
    const size_t HISTOGRAM_ELEMENTS = 1 << B;                                       // Number of elements in shared histogram
    const size_t HISTOGRAM_SIZE     = sizeof(unsigned int) * HISTOGRAM_ELEMENTS;    // Size in bytes of shared histogram
    const size_t TILE_ELEMENTS      = TS * E;                                       // The number of elements processed
    const T      MASK               = (1 << B) - 1;                                 // The mask needed to extract a digit

    // Pointers to additional global device memory needed to store histograms
    unsigned int* d_histogram;
    unsigned int* d_histogram_scan;
    unsigned int* d_histogram_transpose;
    void* d_tmp_storage;
    size_t d_tmp_storage_size;
    
    public:
    void InitMemory(size_t N) {

        const size_t NUM_BLOCKS = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
        const size_t HISTOGRAM_GLOBAL_SIZE   = NUM_BLOCKS * HISTOGRAM_SIZE;

        cudaMalloc((void**)&(this->d_histogram), HISTOGRAM_GLOBAL_SIZE);
        cudaMalloc((void**)&(this->d_histogram_scan), HISTOGRAM_GLOBAL_SIZE);
        cudaMalloc((void**)&(this->d_histogram_transpose), HISTOGRAM_GLOBAL_SIZE);

        cub::DeviceScan::ExclusiveScan(NULL, 
                                       d_tmp_storage_size,
                                       d_histogram, 
                                       d_histogram, 
                                       cub::Sum(), 
                                       0, 
                                       NUM_BLOCKS*HISTOGRAM_ELEMENTS);
        cudaMalloc((void**)&d_tmp_storage, d_tmp_storage_size);
    }

    // Cleanup and free memory
    void Cleanup();

    // The main function that 
    void Sort(T* d_in, T* d_out, const size_t N) {

        const size_t NUM_BLOCKS = (N + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
    
        int iterations = sizeof(T)*8 / B;
        for (int i = 0; i < iterations; i++) {
            localSort<T, E, B, TS, MASK, TILE_ELEMENTS, HISTOGRAM_ELEMENTS><<<NUM_BLOCKS, TS>>>(d_in, d_out, N, d_histogram, i);

            transposeTiled<unsigned int, 32>(d_histogram,
                                             d_histogram_transpose,
                                             NUM_BLOCKS,
                                             HISTOGRAM_ELEMENTS);
            cub::DeviceScan::ExclusiveScan(d_tmp_storage, 
                                           d_tmp_Storage_size, 
                                           d_histogram_transpose, 
                                           d_histogram_scan, 
                                           cub::Sum(),
                                           NUM_BLOCKS * HISTOGRAM_ELEMENTS);
            transposeTiled<unsigned int, 32>(d_histogram_scan, 
                                             d_histogram_transpose, 
                                             HISTOGRAM_ELEMENTS, 
                                             NUM_BLOCKS);

            unsigned int* tmp;
            tmp = d_histogram_scan;
            d_histogram_scan = d_histogram_transpose;
            d_histogram_transpose = tmp;

            swapBuffers<T, E, B, TS, MASK, TILE_ELEMENTS, HISTOGRAM_ELEMENTS><<<NUM_BLOCKS, TS>>>(d_in,
                                                               d_out, 
                                                               N, 
                                                               d_histogram, 
                                                               d_histogram_scan, 
                                                               i // The digit we are looking at
                                                               );
        }
        

    }

};