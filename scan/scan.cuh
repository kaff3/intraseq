#pragma once

#include<cuda_runtime.h>

#define DIV(x,y) ((x+y-1) / y)
#define WARP 32
#define WARP_LOG 5
#define MAX_THREADS_BLOCK 1024
#define BLOCKDIM_X1 1024
#define BLOCKDIM_X2 256

/**
 * Performs a scan within a warp
 * ptr: Pointer to memory for data to be scanned
 * idx: The same as threadIdx.x, but are given as argument such that caller
 *      can use the fucntion with fewer than 32 threads in a warp.
**/
template<
    typename T
>
__device__ inline
T scan_inc_warp(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane = idx & (WARP-1);
    #pragma unroll
    for (int d = 0; d < WARP_LOG; d++){
        unsigned int h = 1 << d;
        if (lane >= h){
            ptr[idx] = ptr[idx-h] + ptr[idx];
        }
    }
    return ptr[idx];
}


/**
 * Performs a scan within a block
 * ptr: Pointer to memory to be scanned
 * idx: The same as threadIdx.x but given as argument to be able to be called with
 *      fewer threads.
*/
template<
    typename T
>
__device__ inline
T scan_inc_block(volatile T* ptr, const unsigned int idx){
    const unsigned int lane = idx & (WARP-1);
    const unsigned int warpid = idx >> WARP_LOG; 

    // All warps perform their scan
    T res = scan_inc_warp(ptr, idx);
    __syncthreads();
    
    // Copy last element of each warp to first warp
    if (lane == (WARP - 1)) ptr[warpid] = res;
    __syncthreads();

    // First warp perform scan
    if (warpid == 0) scan_inc_warp(ptr, idx);
    __syncthreads();

    // Accumulate across warps
    if (warpid > 0) res = res + ptr[warpid-1];
    __syncthreads();

    return res;
}

template<typename T>
__global__
void scan_kernel(T* d_in, size_t N, size_t iter) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // Load into shared memory
    volatile __shared__ T sh_mem[BLOCKDIM_X1 * sizeof(T)];
    if (gid < N)
        sh_mem[tid] = d_in[gid];
    __syncthreads();

    // Do the scan
    for (size_t i = 0; i < iter; i++) {
        sh_mem[tid] = scan_inc_block<T>(sh_mem, tid);
        __syncthreads();
    }

    // Write back result
    if (gid < N)
        d_in[gid] = sh_mem[tid];
}


template<typename T>
__global__
void scan_kernel_seq(T* d_in, size_t N, size_t iter) {
    unsigned int tid = threadIdx.x;

    const size_t num_elems = MAX_THREADS_BLOCK / BLOCKDIM_X2;

    volatile __shared__ T sh_mem[BLOCKDIM_X2 * sizeof(T) * num_elems];
    volatile __shared__ T sh_tmp[BLOCKDIM_X2 * sizeof(T)];

    // Load into shared memory
    #pragma unroll
    for (size_t i = 0; i < num_elems; i++) {
        size_t s_index = threadIdx.x + i * blockDim.x;
        size_t d_index = blockIdx.x * MAX_THREADS_BLOCK + s_index;
        if (d_index < N)
            sh_mem[s_index] = d_in[d_index];
    }

    // Perform iter scans
    for (size_t i = 0; i < iter; i++) {

        // Do element wise scans
        T accum = 0;
        #pragma unroll
        for (size_t i = 0; i < num_elems; i++) {
            size_t offset = num_elems * tid;
            accum += sh_mem[offset + i];
            sh_mem[offset + i] = accum;
        }
        sh_tmp[threadIdx.x] = accum;
        __syncthreads();
        
        sh_tmp[tid] = scan_inc_block<T>(sh_tmp, tid);
        __syncthreads();
        
        if (tid > 0) {
            T accum = sh_tmp[tid-1];
            
            for (size_t i = 0; i < num_elems; i++) {
                size_t offset = num_elems * tid;
                sh_mem[offset + i] += accum;
            }
        }
        __syncthreads();
    }

    // Store in global memory
    #pragma unroll
    for (size_t i = 0; i < num_elems; i++) {
        size_t s_index = threadIdx.x + i * blockDim.x;
        size_t d_index = blockIdx.x * MAX_THREADS_BLOCK + s_index;
        if (d_index < N)
            d_in[d_index] =  sh_mem[s_index];
    }

}
    
template<typename T>
__global__
void scan_kernel_seq_reg(T* d_in, size_t N, size_t iter) {
    unsigned int tid = threadIdx.x;

    const size_t num_elems = MAX_THREADS_BLOCK / BLOCKDIM_X2;

    volatile __shared__ T sh_mem[BLOCKDIM_X2 * sizeof(T) * num_elems];
    volatile __shared__ T sh_tmp[BLOCKDIM_X2 * sizeof(T)];

    // Load into shared memory
    #pragma unroll
    for (size_t i = 0; i < num_elems; i++) {
        size_t s_index = threadIdx.x + i * blockDim.x;
        size_t d_index = blockIdx.x * MAX_THREADS_BLOCK + s_index;
        if (d_index < N)
            sh_mem[s_index] = d_in[d_index];
    }

    // Perform iter scans
    for (size_t i = 0; i < iter; i++) {

        // Do element wise scans
        T accum = 0;
        #pragma unroll
        for (size_t i = 0; i < num_elems; i++) {
            size_t offset = num_elems * tid;
            accum += sh_mem[offset + i];
            sh_mem[offset + i] = accum;
        }
        sh_tmp[threadIdx.x] = accum;
        __syncthreads();
        
        sh_tmp[tid] = scan_inc_block<T>(sh_tmp, tid);
        __syncthreads();
        
        if (tid > 0) {
            T accum = sh_tmp[tid-1];
            
            for (size_t i = 0; i < num_elems; i++) {
                size_t offset = num_elems * tid;
                sh_mem[offset + i] += accum;
            }
        }
        __syncthreads();
    }

    // Store in global memory
    #pragma unroll
    for (size_t i = 0; i < num_elems; i++) {
        size_t s_index = threadIdx.x + i * blockDim.x;
        size_t d_index = blockIdx.x * MAX_THREADS_BLOCK + s_index;
        if (d_index < N)
            d_in[d_index] =  sh_mem[s_index];
    }

}
