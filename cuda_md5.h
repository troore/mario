#ifndef __CUDA_MD5_H__
#define __CUDA_MD5_H__

#define GLOBAL_MEMORY_CAPACITY (1 << 10)

#define MAX_MSG_LEN 4

#define THREADS_PER_BLOCK 100

typedef unsigned int uint;

void print_md5(uint *hash, bool crlf = true);
void md5_prep(char *c0);
double gpu_execute_kernel(char *gpuWords, uint *gpuHashes, int activeThreads);
double cpu_execute_kernel (char *cpuWords, uint *cpuHashes, uint hashSize);
void init_constants();
void md5_cpu(uint w[16], uint &a, uint &b, uint &c, uint &d);
void md5_cpu_v2(const uint *in, uint &a, uint &b, uint &c, uint &d);
__host__ __device__ void md5_pad(char *paddedWord, char *gpuWord, uint len);


#endif
