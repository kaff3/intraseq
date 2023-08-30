#pragma once

#include<cuda_runtime.h>



#define DIV(x,y) ((x+y-1) / y)

/******************************************************************************

******************************************************************************/
// template<typename T>
// __device__
// void mem_load(T* in, T* out, size_t N) {
//     size_t idx = 
// }

template<
    typename T,
    size_t BLOCK_SIZE
>
__global__
void block_reduce(T* d_in, T* d_out, size_t N) {
    // TODO: handle block size not multiple of 2 by padding

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    
    // Read to shared memory
    __shared__ T sh_mem[BLOCK_SIZE * sizeof(T) * 2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        if (gid < N)
            sh_mem[tid + BLOCK_SIZE * i] = d_in[gid + BLOCK_SIZE * i];
    }
    __syncthreads();

    // Do the reduction
    for (size_t stride = BLOCK_SIZE; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_mem[tid] = sh_mem[tid] + sh_mem[tid + stride];
        }
    }

    // write block result
    d_out[blockIdx.x] = sh_mem[0];
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






/******************************************************************************
First kernel for naive scan. Will scan the blocks independently writing out the
last scanned element of each block to d_tmp to be scanned afterwards 
******************************************************************************/
template<typename T>
__global__ 
void scan_naive_kernel_block(T* d_in, T* d_out, T* d_tmp) {

    // All threads read two elements
    size_t idx1 = 2 * threadIdx.x;
    size_t idx2 = idx1 + 1;
    T res = d_in[idx1] + d_in[idx2];

}

/******************************************************************************
The kernel used to add the aggregatee value to the scanned blocks
******************************************************************************/
template<typename T>
__global__
void add_kernel(T* d_tmp, T* d_in) {
    
}
