#pragma once

#include<cuda_runtime.h>

// Kernel A
// Load and sort tile.
// Write histogram and sorted data to global mem
template<
    typename T,     // The type of the array to be sorted 
    int E,          // The number of elements pr. thread
    int B,          // The nmber of bits in a digit
    int TS,         // The nnumber of threads in a block
    int MASK
> __global__ void 
localSort(T* d_in, T* d_out, size_t d_size, unsigned int* d_histogram, int digit) {

    // Specialize cub blockscan
    typedef cub::BlockScan<unsigned int, TS> BlockScan;

    // very important computations
    int last_tile = ((N + TILE_ELEMENTS - 1) / TILE_ELEMENTS) - 1;
    int last_size = N % TILE_ELEMENTS == 0 ? TILE_ELEMENTS : N % TILE_ELEMENTS;
    int local_tile_size = blockIdx.x == last_tile ? last_size : TILE_ELEMENTS;

    // Allocate all the needed shared memory
    __shared__ T s_tile[];
    __shared__ unsigned int s_histogram[];
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
        if (d_index < d_size) {
            tile[s_index] = d_in[d_index];
        }
    }

    // B iterations of 1 bit splits
    for (int i = 0; i < B; i++) {
        // Load the elements into local mem
        #pragma unroll
        for (int j = 0; j < E; j++) {
            size_t index = threadIdx.x * E + j;
            if (index > d_size) {
                elements[i] = tile[index];
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
        BlocksScan(ps_storage.ps0).ExclusiveScan(ps0, ps0, cub::Sum(), aggregate);
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
            T elmDigit = GET_DIGIT(elemtns[i], digit*B, mask);
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

// Kernel B
// Scan global histogram
__global__ void histogramScan();

// Kernel C
// Copy elements to correct output
__global__ void swapBuffers();

/* 
Kan man ikke gøre det hele i en enkelt kernel ved at bruge nogle smarte CUB ting til at skanne
det "globale" histogram på tværs af thread blocks?
*/