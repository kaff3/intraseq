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
    
    T tmp;
    if (lane == (WARP - 1)) tmp = ptr[idx];
    __syncthreads();

    if (lane == (WARP - 1)) ptr[idx] = tmp;
    __syncthreads();
    
    // Copy last element of each warp to first warp
    // if (lane == (WARP - 1)) ptr[warpid] = res;
    // __syncthreads();

    // First warp perform scan
    if (warpid == 0) scan_inc_warp(ptr, idx);
    __syncthreads();

    // Accumulate across warps
    if (warpid > 0) res = res + ptr[warpid-1];
    
    return res;
}


/**
 * Reduces a given array, producing a reduced value pr. block placed in d_out
 * d_in: Pointer to the data to be reduced. Has length N
 * d_out: Pointer to where the pr. block reduced values should be stored
 *        Has length gridDim.x
 * n: The number of elements in d_in;
*/
template<
    typename T
>
__global__
void reduce_kernel(T* d_in, T* d_out, size_t N) {
    // TODO: handle block size not multiple of 2 by padding

    // Will have size blockDim.x * sizeof(T)
    extern volatile __shared__ T sh_mem[];

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    
    // Load into shared mem
    if (tid < N) sh_mem[tid] = d_in[gid];
    __syncthreads();

    // Do a scan. Will only need the value in the last thread
    T res = scan_inc_block(sh_mem, tid);

    // Last thread write out the reduced value
    if (tid == (blockDim.x - 1))
        d_out[blockIdx.x] = res;
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

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
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
            sh_mem[offset + i] += accum;
            accum = sh_mem[i + offset];
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
    





/**
 * This function orchastrates the the calls and synchronization between
 * kernels needed to perform a scan. 
*/
template<typename T>
void scan() {
    // TODO:
    /**
     * Should the function just take host ararys as arguments and then copy
     * them to device or should it take device arrays?
    */
}









/******************************************************************************
This is the function to call from the CPU side to do the scan. It will call the
kernel on the correct input
******************************************************************************/
// template<
//     typename T,     // The type of the array
//     size_t B,       // The number of elements pr block
//     size_t BLOCK_SIZE
// >
// void scan_naive(T* d_in, T* d_out, T* d_tmp, size_t N) {

//     // Compute number of blocks needed for reduce
//     size_t outer_num_blocks = DIV(N, BLOCK_SIZE);
//     reduce_kernel<<<outer_num_blocks, BLOCK_SIZE>>>();

//     size_t inner_num_blocks = DIV(outer_num_blocks, BLOCK_SIZE);
//     scan_kernel<<<inner_num_blocks, BLOCK_SIZE>>>();
    
//     scan_kernel<<<outer_num_blocks, BLOCK_SIZE>>>();

// }



    // // Allocate device memory
    // // TODO: Can probably be more elegant
    // size_t mem;
    // if ((N % B) == 0)
    //     mem = N * sizeof(T);
    // else
    //     mem = (N - (N % B) + B) * sizeof(T)

    // T* d_in = cudaMalloc(mem);
    // T* d_out = cudaMalloc(mem);
    // // TODO: Copy to device

    // // Compute the number of blocks and threads needed 
    // size_t num_blocks = DIV(N, B);
    // size_t num_threads = DIV(B, 2);

    // // Additional memory for intermediate sizes
    // T* d_tmp1 = cudaMalloc(num_blocks * sizeof(T));
    // T* d_tmp2 = cudaMalloc(num_blocks * sizeof(T));

    // // Invoke kernel
    // scan_naive_kernel_block<<<num_threads, num_blocks>>>(d_in, d_out, d_tmp1);

    // // Scan the tmp array
    // scan_naive_kernel_block<<<?, ?>>>(d_tmp1, d_tmp2, NULL);

    // // Add values to scanned array

