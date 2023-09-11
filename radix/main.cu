#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<limits>

#include"radix.cuh"
#include"../shared/helper.cu.h"
#include"../shared/timing.h"

int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Usage: ./radix <gpu runs>\n");
        return 0;
    }

    int gpu_runs = atoi(argv[1]);

    std::vector<size_t> sizes = {
        1 << 5,
        1 << 6,
        1 << 7,
        1 << 8,
        1 << 9,
        1 << 10,
        1 << 11,
        1 << 12,
        1 << 13,
        1 << 14,
        1 << 15,
        1 << 16,
    };
        
       

    for(int i = 0; i < sizes.size(); i++) {
        // Specialize the radix sort algorithm
        Radix<unsigned int, 256, 4, 4> radix4;
        radix4.InitMemory(sizes[i]);


        // Allocate host memory
        unsigned int* h_in  = (unsigned int*)malloc(sizeof(unsigned int) * sizes[i]);
        unsigned int* h_out = (unsigned int*)malloc(sizeof(unsigned int) * sizes[i]);

        randomInitNat<unsigned int>(h_in, sizes[i], UINT_MAX);

        // move to device
        unsigned int* d_in;
        unsigned int* d_out;
        cudaMemcpy(d_in, (void*)h_in, sizeof(unsigned int) * sizes[i], cudaMemcpyHostToDevice);

        Timer t1;

        t1.Start();
        radix4.Sort(d_in, d_out, sizes[i]);
        t1.Stop();

        printf("%u\n", t1.Get());

        cudaMemcpy((void*)h_out, d_in, sizeof(unsigned int) * sizes[i], cudaMemcpyDeviceToHost);
        
        #ifdef DO_VALIDATE
        // Allocations needed for validation
        unsigned int* h_out_cub = (unsigned int*)malloc(sizeof(unsigned int) * sizes[i])
        void* d_tmp_storage_cub = NULL;
        size_t tmp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_tmp_storage_cub, tmp_storage_bytes, d_in, d_out, N);
        cudaMalloc(&d_tmp_storage_cub, tmp_storage_bytes);

        // Copy to device
        cudaMemcpy(d_in, (void*)h_in, sizeof(unsigned int) * sizes[i], cudaMemcpyHostToDevice);
        cub::DeviceRadixSort::SortKeys(d_tmp_storage, tmp_storage_bytes, d_in, d_out, sizes[i]);
        cudaMemcpy((void*)h_in, d_out, sizeof(unsigned int) * sizes[i], cudaMemcpyDeviceToHost);
        
        // Do the validation
        bool valid = true;
        for(size_t j = 0; j < sizes[i]; j++) {
            if(h_out[i] != h_out_cub[i]) {
                printf("Not Valid at j=%u\n", j);
                valid = false;
                break;
            }
        }
        cudaFree(d_tmp_storage_cub);
        free(h_out_cub);
        #endif



        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        radix4.Cleanup();
    }

    return 0;
}