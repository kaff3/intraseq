#ifndef RADIX_FUT
#define RADIX_FUT

#include<vector>
#include<cuda_runtime.h>

#include "./radix-fut.h"  // automagically generated
#include "./helper.cu.h"
#include "./timing.h"


struct RadixFut
{
private:
    futhark_context_config* cfg8;
    futhark_context_config* cfg16;
    futhark_context_config* cfg32;
    futhark_context_config* cfg64;

    futhark_context* ctx8;
    futhark_context* ctx16;
    futhark_context* ctx32;
    futhark_context* ctx64;

    futhark_u8_1d*  in8;  
    futhark_u8_1d*  out8;
    futhark_u16_1d* in16;  
    futhark_u16_1d* out16;
    futhark_u32_1d* in32; 
    futhark_u32_1d* out32; 
    futhark_u64_1d* in64;  
    futhark_u64_1d* out64; 

public:

    RadixFut() {
        this->cfg8  = futhark_context_config_new();
        this->cfg16 = futhark_context_config_new();
        this->cfg32 = futhark_context_config_new();
        this->cfg64 = futhark_context_config_new();

        this->ctx8  = futhark_context_new(this->cfg8);
        this->ctx16 = futhark_context_new(this->cfg16);
        this->ctx32 = futhark_context_new(this->cfg32);
        this->ctx64 = futhark_context_new(this->cfg64);

    }

    ~RadixFut() {
        futhark_context_free(this->ctx8);
        futhark_context_free(this->ctx16);
        futhark_context_free(this->ctx32);
        futhark_context_free(this->ctx64);

        futhark_context_config_free(this->cfg8);
        futhark_context_config_free(this->cfg16);
        futhark_context_config_free(this->cfg32);
        futhark_context_config_free(this->cfg64);
    }


    void Bench(std::vector<int> sizes) {

        for (int i = 0; i < sizes.size(); i++) {

            long N = (long)sizes[i];

            size_t size8  = sizeof(unsigned char) * N;
            size_t size16 = sizeof(unsigned short) * N;
            size_t size32 = sizeof(unsigned int) * N;
            size_t size64 = sizeof(unsigned long) * N;

            // Host allocations
            unsigned char*   h_in8  = (unsigned char*)  malloc(size8);
            unsigned short*  h_in16 = (unsigned short*) malloc(size16);
            unsigned int*    h_in32 = (unsigned int*)   malloc(size32);
            unsigned long*   h_in64 = (unsigned long*)  malloc(size64);

            randomInitNat<unsigned char>(h_in8, N, N);
            randomInitNat<unsigned short>(h_in16, N, N);
            randomInitNat<unsigned int>(h_in32, N, N);
            randomInitNat<unsigned long>(h_in64, N, N);

            unsigned char* d_in8;
            unsigned short* d_in16;
            unsigned int* d_in32;
            unsigned long* d_in64;
            cudaMalloc((void**)&d_in8, size8);
            cudaMalloc((void**)&d_in16, size16);
            cudaMalloc((void**)&d_in32, size32);
            cudaMalloc((void**)&d_in64, size64);

            // Setup the data for each context
            futhark_u8_1d*  fut_in8  = futhark_new_u8_1d(this->ctx8, d_in8, N);
            futhark_u16_1d* fut_in16 = futhark_new_u16_1d(this->ctx16, d_in16, N);
            futhark_u32_1d* fut_in32 = futhark_new_u32_1d(this->ctx32, d_in32, N);
            futhark_u64_1d* fut_in64 = futhark_new_u64_1d(this->ctx64, d_in64, N);


            unsigned char* d_out8;
            unsigned short* d_out16;
            unsigned int* d_out32;
            unsigned long* d_out64;
            cudaMalloc((void**)&d_out8, size8);
            cudaMalloc((void**)&d_out16, size16);
            cudaMalloc((void**)&d_out32, size32);
            cudaMalloc((void**)&d_out64, size64);

            // Setup the data for each context
            futhark_u8_1d*  fut_out8  = futhark_new_u8_1d(this->ctx8, d_out8, N);
            futhark_u16_1d* fut_out16 = futhark_new_u16_1d(this->ctx16, d_out16, N);
            futhark_u32_1d* fut_out32 = futhark_new_u32_1d(this->ctx32, d_out32, N);
            futhark_u64_1d* fut_out64 = futhark_new_u64_1d(this->ctx64, d_out64, N);
            
            // Dry runs;
            futhark_entry_sort_u8(this->ctx8, &fut_out8, fut_in8);
            cudaDeviceSynchronize();
            // futhark_entry_sort_u16(this->ctx16, &fut_out16, fut_in16);
            // cudaDeviceSynchronize();
            // futhark_entry_sort_u32(this->ctx32, &fut_out32, fut_in32);
            // cudaDeviceSynchronize();
            // futhark_entry_sort_u64(this->ctx64, &fut_out64, fut_in64);
            // cudaDeviceSynchronize();

            Timer t8, t16, t32, t64;

            t8.Start();
            futhark_entry_sort_u8(this->ctx8, &fut_out8, fut_in8);
            cudaDeviceSynchronize();
            t8.Stop();

            t16.Start();
            futhark_entry_sort_u16(this->ctx16, &fut_out16, fut_in16);
            cudaDeviceSynchronize();
            t16.Stop();

            t32.Start();
            futhark_entry_sort_u32(this->ctx32, &fut_out32, fut_in32);
            cudaDeviceSynchronize();
            t32.Stop();

            t64.Start();
            futhark_entry_sort_u64(this->ctx64, &fut_out64, fut_in64);
            cudaDeviceSynchronize();
            t64.Stop();

            printf("Fut u8:  %i\n", t8.Get());
            printf("Fut u16: %i\n", t16.Get());
            printf("Fut u32: %i\n", t32.Get());
            printf("Fut u64: %i\n", t64.Get());

            free(h_in8);
            free(h_in16);
            free(h_in32);
            free(h_in64);

            futhark_free_u8_1d(this->ctx8, fut_in8);
            futhark_free_u16_1d(this->ctx16, fut_in16);
            futhark_free_u32_1d(this->ctx32, fut_in32);
            futhark_free_u64_1d(this->ctx64, fut_in64);

            futhark_free_u8_1d(this->ctx8, fut_out8);
            futhark_free_u16_1d(this->ctx16, fut_out16);
            futhark_free_u32_1d(this->ctx32, fut_out32);
            futhark_free_u64_1d(this->ctx64, fut_out64);



        }

    }



};





#endif