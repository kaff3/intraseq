

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "cub/cub.cuh"


#define THREAD_ELEMENTS     4   // 
#define NUM_THREADS         256
#define TILE_SIZE           (NUM_THREADS * THREAD_ELEMENTS)
#define B                   4
#define HISTOGRAM_SIZE      16  // 2^B
#define GET_DIGIT(V, I, M)  ((V >> I) & M)

// #define NUM_POSITIONS       (NUM_THREADS * 2)


//========================================
// ====== Helper device functions =======
//========================================


__device__ void loadTile(unsigned int* d_in, uint64_t input_arr_size, unsigned int* tile) {
    #pragma unroll
    for (int i = 0; i < THREAD_ELEMENTS; i++) {
        unsigned int s_index = threadIdx.x + i * blockDim.x;
        unsigned int d_index = blockIdx.x * TILE_SIZE + s_index;
        if (d_index < input_arr_size) {
            tile[s_index] = d_in[d_index];
        }
    }
}

__device__ void loadThreadElements(unsigned int* elements, uint64_t tile_size, unsigned int* tile) {
    #pragma unroll
    for (int j = 0; j < THREAD_ELEMENTS; j++) {
        unsigned index = threadIdx.x * THREAD_ELEMENTS + j;
        if (index < tile_size) {
            elements[j] = tile[index];
        }
    }
}

__device__ void countBits(unsigned int* elements, unsigned int* ps0, unsigned int* ps1, 
                          uint64_t tile_size, int curr_digit, int i) {
    #pragma unroll
    for (int j = 0; j < THREAD_ELEMENTS; j++) {
        unsigned int index = threadIdx.x * THREAD_ELEMENTS + j;
        if (index < tile_size) {
            unsigned int bit = GET_DIGIT(elements[j], curr_digit*B+i, 0x1);
            *ps0 += (bit == 0 ? 1 : 0);
            *ps1 += (bit == 1 ? 1 : 0);
        }
    }
}

__device__ void countDigits(unsigned int* elements, unsigned int* histo, uint64_t tile_size, int curr_digit) {
    #pragma unroll
    for (int j = 0; j < THREAD_ELEMENTS; j++) {
        unsigned int index = threadIdx.x * THREAD_ELEMENTS + j;
        if (index < tile_size) {
            unsigned int val = GET_DIGIT(elements[j], curr_digit*B, 0xF);
            histo[val] += 1;
        }
    }
}

__device__ void scanPositions(unsigned int* ps0, unsigned int* ps1, unsigned int init) {
    typedef cub::BlockScan<unsigned int, NUM_THREADS> BlockScan;

    __shared__ union {
        typename BlockScan::TempStorage ps0;
        typename BlockScan::TempStorage ps1;
    } ps_storage;
    unsigned int aggregate;

    BlockScan(ps_storage.ps0).ExclusiveScan(*ps0, *ps0, init, cub::Sum(), aggregate);
    BlockScan(ps_storage.ps1).ExclusiveScan(*ps1, *ps1, aggregate, cub::Sum());
}


__global__ void kernel12(unsigned int* d_in, unsigned int* d_out, uint64_t input_arr_size, 
                         unsigned int* d_histogram, int curr_digit){

    // Load to shared memory
    __shared__ unsigned int s_tile[TILE_SIZE];
    loadTile(d_in, input_arr_size, s_tile);
    __syncthreads();

    // find the size of the local tile for bounds checking
    unsigned int last_tile_id = ((input_arr_size + TILE_SIZE - 1) / TILE_SIZE) - 1;
    unsigned int size_of_last_tile = input_arr_size % TILE_SIZE;
    unsigned int tile_id = blockIdx.x;
    size_of_last_tile = size_of_last_tile == 0 ? TILE_SIZE : size_of_last_tile;
    unsigned int local_tile_size = tile_id == last_tile_id ? size_of_last_tile : TILE_SIZE;

    // Sort in shared memory b iterations of 1-bit split
    unsigned int elements[THREAD_ELEMENTS];
    for (int i = 0; i < B; i++) {

        // Read elements
        loadThreadElements(elements, local_tile_size, s_tile);
        // __syncthreads();

        // Count bits
        unsigned int ps0 = 0;
        unsigned int ps1 = 0;
        countBits(elements, &ps0, &ps1, local_tile_size, curr_digit, i);
        __syncthreads();

        // Perform a blockwise scan across threads
        scanPositions(&ps0, &ps1, 0);

        // Sort by scattering
        #pragma unroll
        for (int j = 0; j < THREAD_ELEMENTS; j++) {
            unsigned int index = threadIdx.x * THREAD_ELEMENTS + j;
            if (index < local_tile_size) {
                unsigned int bit = GET_DIGIT(elements[j], curr_digit*B+i, 0x1);
                unsigned int pos = (bit == 0 ? ps0 : ps1);
                ps0 += (bit == 0 ? 1 : 0);
                ps1 += (bit == 1 ? 1 : 0);
                s_tile[pos] = elements[j];
            }
        }
        __syncthreads(); // For next iteration

    } // Big loop end


    // zero initialize histogram
    __shared__ unsigned int s_histogram[HISTOGRAM_SIZE];
    if (threadIdx.x < HISTOGRAM_SIZE){
        s_histogram[threadIdx.x] = 0;
    }
    __syncthreads();


    // Compute final histogram
    #pragma unroll
    for (int i = 0; i < THREAD_ELEMENTS; i++) {
        unsigned int index = threadIdx.x * THREAD_ELEMENTS + i;
        if (index < local_tile_size) {
            unsigned int digit = GET_DIGIT(elements[i], curr_digit*B, 0xF); // TODO: Fix mask somehow
            atomicAdd(s_histogram + digit, 1);
        }
    }
    __syncthreads();


    // Write histogram to global memory transposed
    if (threadIdx.x < HISTOGRAM_SIZE) {
        d_histogram[gridDim.x * threadIdx.x + blockIdx.x] = s_histogram[threadIdx.x];
        // d_histogram[gridDim.x * blockIdx.x + threadIdx.x] = s_histogram[threadIdx.x];
    }


    // Write sorted tile back to global memory. coalesced
    #pragma unroll
    for (int i = 0; i < THREAD_ELEMENTS; i++) {
        unsigned int s_index = threadIdx.x + blockDim.x * i;
        unsigned int d_index = blockIdx.x * TILE_SIZE + s_index;
        if (d_index < input_arr_size) {
            d_out[d_index] = s_tile[s_index];
        }
    }
}

__global__ void globalScatter(unsigned int* d_in, unsigned int* d_out, unsigned int input_arr_size, 
                              unsigned int* d_histogram, int curr_digit) {
    // find the size of the local tile for bounds checking
    unsigned int last_tile_id = ((input_arr_size + TILE_SIZE - 1) / TILE_SIZE) - 1;
    unsigned int size_of_last_tile = input_arr_size % TILE_SIZE;
    size_of_last_tile = size_of_last_tile == 0 ? TILE_SIZE : size_of_last_tile;
    unsigned int tile_id = blockIdx.x;
    unsigned int local_tile_size = tile_id == last_tile_id ? size_of_last_tile : TILE_SIZE;

    
    // Read memory to shared first again
    __shared__ unsigned int s_tile[TILE_SIZE];
    loadTile(d_in, input_arr_size, s_tile);
    __syncthreads();

    unsigned int elements[THREAD_ELEMENTS];
    loadThreadElements(elements, local_tile_size, s_tile);
    __syncthreads();

    // Read the now scanned histogram back into shared such that we have faster
    // access to it and can update it faster. As the histogram were written to global
    // transposed, it cannot be read back in coalesced.
    __shared__ unsigned int s_histogram[HISTOGRAM_SIZE];
    if (threadIdx.x < HISTOGRAM_SIZE) {
        unsigned int d_index = gridDim.x * threadIdx.x + blockIdx.x;
        s_histogram[threadIdx.x] = d_histogram[d_index];
    }
    __syncthreads();

    // positions for each digit
    unsigned int ps[HISTOGRAM_SIZE] = {0};
    __syncthreads();

    countDigits(elements, ps, local_tile_size, curr_digit);
    __syncthreads();

    typedef cub::BlockScan<unsigned int, NUM_THREADS> BlockScan;

    // unsigned int aggregate = 0;
    
    for (int i = 0; i < HISTOGRAM_SIZE; i++){
        __shared__ typename BlockScan::TempStorage ps_tmp;
        unsigned int offset = s_histogram[i];
        BlockScan(ps_tmp).ExclusiveScan(ps[i], ps[i], offset, cub::Sum());
    }
    __syncthreads();   


    // Can now perform the scatter back into global memory. This is going to be done
    // in a strided way.
    for (int i = 0; i < THREAD_ELEMENTS; i++) {
        // unsigned int s_index = threadIdx.x + i * blockDim.x;
        unsigned int s_index = threadIdx.x * THREAD_ELEMENTS + i;
        unsigned int d_index = blockIdx.x * TILE_SIZE + s_index;
        if (d_index < input_arr_size) {
            unsigned int val   = elements[i];
            unsigned int digit = GET_DIGIT(val, curr_digit*B, 0xF);
            // unsigned int old   = atomicAdd(ps + digit, 1);
            unsigned int old = ps[digit];
            ps[digit] += 1;
            d_out[old] = val;
        }
    }
}



// form sorting_test.cu
void randomInitNat(unsigned int* data, const unsigned int size, const unsigned int H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

int main(int argc, char* argv[]){

    if (argc < 2) { printf("Hey, don't you think it would be smart to give me an argument?\n"); return 0; }

    const uint64_t N = atoi(argv[1]);
    // TODO: maybe check N if it is too big
    const uint64_t arr_size = N * sizeof(unsigned int);

    // Host allocations
    unsigned int* h_in  = (unsigned int*) malloc(arr_size);
    unsigned int* h_out = (unsigned int*) malloc(arr_size);

    // Create random array to sort
    randomInitNat(h_in, N, N);

    // Compute blocks and block sizes
    unsigned int num_blocks = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Device allocations
    unsigned int* d_in;
    unsigned int* d_out;
    unsigned int* d_histogram;
    unsigned int* d_histogram_scanned;
    cudaMalloc((void**)&d_in,  arr_size);
    cudaMalloc((void**)&d_out, arr_size);
    cudaMalloc((void**)&d_histogram, num_blocks * HISTOGRAM_SIZE * sizeof(unsigned int));
    cudaMalloc((void**)&d_histogram_scanned, num_blocks * HISTOGRAM_SIZE * sizeof(unsigned int));

    // Copy initial array to device
    cudaMemcpy(d_in, h_in, arr_size, cudaMemcpyHostToDevice);


    printf("num blocks:  %i\n", num_blocks);
    printf("num threads: %i\n", num_blocks*NUM_THREADS);
    printf("tile size:   %i\n", TILE_SIZE);


    printf("Start\n");
    for (int i = 0; i < (sizeof(unsigned int)*8)/B; i++) {
        // kernel 1 + 2
        kernel12<<< num_blocks, NUM_THREADS >>>(d_in, d_out, N, d_histogram, i);
        // cudaDeviceSynchronize();
    

        // Perform device wide scan over histogram (kernel 3)
        void* d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        // cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes, d_histogram, d_histogram_scanned, HISTOGRAM_SIZE*num_blocks);
        cub::DeviceScan::ExclusiveScan(d_tmp_storage, tmp_storage_bytes, d_histogram, d_histogram_scanned, cub::Sum(), 0, HISTOGRAM_SIZE*num_blocks);
        // cudaDeviceSynchronize();
        cudaMalloc(&d_tmp_storage, tmp_storage_bytes);
        // cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes, d_histogram, d_histogram_scanned, HISTOGRAM_SIZE*num_blocks);
        cub::DeviceScan::ExclusiveScan(d_tmp_storage, tmp_storage_bytes, d_histogram, d_histogram_scanned, cub::Sum(), 0, HISTOGRAM_SIZE*num_blocks);
        cudaFree(d_tmp_storage); // TODO: Should we free this or does it happen automatically?
        // cudaDeviceSynchronize();

        // kernel 4
        globalScatter<<< num_blocks, NUM_THREADS >>>(d_out, d_in, N, d_histogram_scanned, i);
        cudaMemset(d_histogram_scanned, 0, sizeof(unsigned int)*HISTOGRAM_SIZE*num_blocks);
        // cudaDeviceSynchronize();

        // Swap input input and output. This is not needed when final kernel is implemented
        // unsigned int* tmp;
        // tmp = d_in;
        // d_in  = d_out;
        // d_out = tmp;
    }
    cudaDeviceSynchronize();
    printf("Stop\n");


    // Copy from device to print result
    cudaMemcpy(h_out, d_in, arr_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < N; i++) {
    //     printf("%10x      %10x\n", h_out[i], h_in[i]);
    // }

    // Clean up memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_histogram);
    cudaFree(d_histogram_scanned);
    free(h_in);
    free(h_out);

    return 0;
}
