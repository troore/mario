#ifndef __CUDA_MD5_H__
#define __CUDA_MD5_H__

#define GLOBAL_MEMORY_CAPACITY (1 << 10)

#define MAX_MSG_LEN 4

#define THREADS_PER_BLOCK 100

typedef unsigned int uint;

//
// structures to store hash values
//
union md5hash
{
	uint ui[4];
	char ch[16];
};

union md5hash_v2
{
	uint ui[5];
	char ch[20];
};

void print_md5(uint *hash, bool crlf = true);
double gpu_execute_kernel(char *gpuWords, uint *gpuHashes, int activeThreads, int level);
double cpu_execute_kernel (char *cpuWords, uint *cpuHashes, uint hashSize);
void md5_cpu(const uint *in, uint &a, uint &b, uint &c, uint &d);
__host__ __device__ void md5_pad(char *paddedWord, char *gpuWord, uint len);
void setup_md5_funcs ();


#endif
