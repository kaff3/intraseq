#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define NUM_THREADS 256
#define TILE_SIZE 1024
#define B 4
#define get_digit(V, I) (V & (0xF << (I * 4)))
#define HISTOGRAM_SIZE 16
#define WARP 32

// ******************** SCAN FROM ASSIGNMENT 2 ******************** 
// __device__ inline typename OP::RedElTp
// scanIncWarp( volatile typename OP::RedElTp* ptr, const unsigned int idx ) {
    
//     const unsigned int lane = idx & (WARP-1);
// #ifdef OLD

//     if(lane==0) {
//         #pragma unroll
//         for(int i=1; i<WARP; i++) {
//             ptr[idx+i] = OP::apply(ptr[idx+i-1], ptr[idx+i]);
//         }
//     }

// #else
    
//     // k = lg(32) = 5
//     #pragma unroll
//     for (int d = 0; d < 5; d++){
//         int h = 1;
//         for (int k = 0; k < d; k++) {
//             h *= 2;
//         }
//         if (lane >= h){
//             ptr[idx] = OP::apply(ptr[idx-h], ptr[idx]);
//         }
//     }        
// #endif

//     return OP::remVolatile(ptr[idx]);
// }


// __device__ inline typename OP::RedElTp
// scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
//     const unsigned int lane   = idx & (WARP-1);
//     const unsigned int warpid = idx >> lgWARP;

//     // 1. perform scan at warp level
//     typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
//     __syncthreads();

//     // 2. place the end-of-warp results in
//     //   the first warp. This works because
//     //   warp size = 32, and 
//     //   max block size = 32^2 = 1024

//     typename OP::RedElTp tmp;
//     if (lane == (WARP-1)) { tmp = OP::remVolatile(ptr[idx]); }
//     __syncthreads();

//     if (lane == (WARP-1)) { ptr[warpid] = tmp; } 
//     __syncthreads();

//     // 3. scan again the first warp
//     if (warpid == 0) scanIncWarp<OP>(ptr, idx);
//     __syncthreads();

//     // 4. accumulate results from previous step;
//     if (warpid > 0) {
//         res = OP::apply(ptr[warpid-1], res);
//     } 

//     return res;
// }
// ******************** SCAN FROM ASSIGNMENT 2 ******************** 


__global__ void kernel12(unsigned int* d_out, unsigned int* d_in, unsigned int arr_size, unsigned int* d_histogram, int curr_digit){
    __shared__ unsigned int s_tile[TILE_SIZE];
    __shared__ unsigned int s_tile_sorted[TILE_SIZE];
    __shared__ unsigned int s_histogram[HISTOGRAM_SIZE];
    int iterations = TILE_SIZE / NUM_THREADS; 
    int idx = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x; 

    // zero initialize histogram
    if (idx < HISTOGRAM_SIZE){
        s_histogram[idx] = 0;
    }

    for (int i = 0; i < iterations; i++){
        unsigned int arr_i = gid + i * NUM_THREADS;
        if (arr_i < arr_size)
            unsigned int val = d_in[arr_i];
        
        // increment histogram
        unsigned int digit = get_digit(val, curr_digit);
        atomicAdd(s_histogram + digit, 1);
        
        // copy to shared memory
        s_tile[idx + i * NUM_THREADS] = val;
    }
    __syncthreads();
    
    // copy local histogram to global memory transposed
    // Because there are 8 warps per block, if for each Warp
    // the two first threads are used, there is better
    // coalesed access.
    const unsigned int lane = idx & (WARP-1);
    if (lane == 0 || lane == 1){
        unsigned int curr_warp = idx / WARP;
        int histogram_i = curr_warp * 2 + lane;
        // p = num blocks = gridDim.x
        d_histogram[gridDim.x * histogram_i + blockIdx.x] = s_histogram[histogram_i];
    }

    // exclusive scan over histogram
    // TODO: currently sequential on a single thread
    if (idx == 0){
        for (int i = HISTOGRAM_SIZE-1; i > 0; i--){
            s_histogram[i] = s_histogram[i-1];
        }
        s_histogram[0] = 0;
        for (int i = 1; i < HISTOGRAM_SIZE; i++){
            s_histogram[i] += s_histogram[i-1];
        }
    }

    for (int i  = 0; i < iterations; i++){
        // foreach val in s_tile
        unsigned int val = s_tile[idx + i * NUM_THREADS];
        unsigned int old = atomicAdd(s_histogram + val, 1);
        s_tile_sorted[old] = val;
    }

    __syncthreads();


    // SKAL VI SKRIVE DET SORTEREDE TILBAGE?
    // KAN MAN IKKE BARE LADE DET LIGGE I SHARED OG VENTE TIL STEP 4?
    for (int i = 0; i < iterations; i++){
        d_out[gid + i * NUM_THREADS] = s_tile_sorted[idx + i * NUM_THREADS]
    }
}

// 
__global__ void kernel3(unsigned int* d_out, unsigned int* d_in, unsigned int* d_histogram){
    
}

__global__ void kernel4(unsigned int* d_out, unsigned int* d_in, unsigned int* d_histogram){

}

int main(){
    unsigned int* vals = (unsigned int*) malloc(10000 * sizeof(unsigned int));


}
