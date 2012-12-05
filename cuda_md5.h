#ifndef __CUDA_MD5_H__
#define __CUDA_MD5_H__

#define GLOBAL_MEMORY_CAPACITY (1 << 10)
typedef unsigned int uint;

void print_md5(uint *hash, bool crlf = true);
void md5_prep(char *c0);
double gpu_execute_kernel(int blocks_x, int blocks_y, int threads_per_block, int shared_mem_required, int realthreads, char *gpuWords, uint *gpuHashes, uint maxWordLen);
double cpu_execute_kernel (char *cpuWords, uint *cpuHashes, uint hashSize, uint maxWordLen);
void init_constants();
void md5_cpu(uint w[16], uint &a, uint &b, uint &c, uint &d);
void md5_cpu_v2(const uint *in, uint &a, uint &b, uint &c, uint &d);
__host__ __device__ void md5_pad(char *paddedWord, char *gpuWord, uint len);


#endif
