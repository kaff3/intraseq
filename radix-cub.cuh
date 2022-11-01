#ifndef PMPH_RADIX_CUB
#define PMPH_RADIX_CUB

#include<cuda_runtime.h>
#include"cub/cub.cuh"

template<typename T>
void RadixSortCub(T* d_in, T* d_out, int N) {
    void* d_tmp_storage = NULL;
    size_t tmp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tmp_storage, tmp_storage_bytes, d_in, d_out, N);
    cudaMalloc(&d_tmp_storage, tmp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_tmp_storage, tmp_storage_bytes, d_in, d_out, N);
    cudaFree(d_tmp_storage);
} 


#endif