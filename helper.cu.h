#ifndef HISTO_HELPER
#define HISTO_HELPER

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <sstream>
#include <string>

#define GPU_RUNS    400

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

#define cudaCheckError() {                                              \
    cudaError_t e=cudaGetLastError();                                   \
    if(e!=cudaSuccess) {                                                \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);                                                          \
    }                                                                   \
}

#define cudaSucceeded(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    std::cerr << "cudaAssert failed: "
              << cudaGetErrorString(code)
              << file << ":" << line
              << std::endl;
    if (abort) {
      exit(code);
    }
  }
}

inline uint32_t ceilLog2(uint32_t H) {
    if (H == 0) { printf("Log2(0) is illegal. Exiting!\n"); exit(1); }
    uint32_t log2_val = 0, pow2_val = 1;
    while(pow2_val < H) {
        log2_val ++;
        pow2_val *= 2;
    }
    return log2_val;
}

void writeRuntimes(std::vector<int> sizes, std::vector<float> our, std::vector<float> cub, const char* fname) {
    std::stringstream ss;
    for (size_t i = 0; i < sizes.size(); i++) {
        if (i != 0)
            ss << ",";
        ss << sizes[i];
    }
    ss << "\n";
    for (size_t i = 0; i < our.size(); i++) {
        if (i != 0)
            ss << ",";
        ss << our[i];
    }
    ss << "\n";
    for (size_t i = 0; i < cub.size(); i++) {
        if (i != 0)
            ss << ",";
        ss << cub[i];
    }
    std::string s = ss.str();

    FILE* f = fopen(fname, "w");
    assert(f != NULL);
    fprintf(f, "%s", s.c_str());
    fclose(f);

}

template<typename T>
void randomInitNat(T* data, const size_t size, const size_t H) {
    for (size_t i = 0; i < size; i++) {
        T r = rand();
        data[i] = r % H;
    }
}

#endif // HISTO_HELPER
