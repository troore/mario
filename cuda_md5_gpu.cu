// CUDA MD5 hash calculation implementation.

#include <stdio.h>
#include "cuda_md5.h"

typedef unsigned int uint;

__device__ inline uint &getw(uint *w, const int i)
{
	return w[i];
}

__device__ inline uint getw(const uint *w, const int i)	// const- version
{
	return w[i];
}


//////////////////////////////////////////////////////////////////////////////
/////////////       Ron Rivest's MD5 C Implementation       //////////////////
//////////////////////////////////////////////////////////////////////////////


/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z))) 

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }


/* Basic MD5 step. Transform buf based on in.
 */
void inline __device__ md5_cuda(const uint *in, uint &a, uint &b, uint &c, uint &d)
{
	const uint a0 = 0x67452301;
	const uint b0 = 0xEFCDAB89;
	const uint c0 = 0x98BADCFE;
	const uint d0 = 0x10325476;

	//Initialize hash value for this chunk:
	a = a0;
	b = b0;
	c = c0;
	d = d0;

  /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
  FF ( a, b, c, d, getw(in,  0), S11, 3614090360); /* 1 */
  FF ( d, a, b, c, getw(in,  1), S12, 3905402710); /* 2 */
  FF ( c, d, a, b, getw(in,  2), S13,  606105819); /* 3 */
  FF ( b, c, d, a, getw(in,  3), S14, 3250441966); /* 4 */
  FF ( a, b, c, d, getw(in,  4), S11, 4118548399); /* 5 */
  FF ( d, a, b, c, getw(in,  5), S12, 1200080426); /* 6 */
  FF ( c, d, a, b, getw(in,  6), S13, 2821735955); /* 7 */
  FF ( b, c, d, a, getw(in,  7), S14, 4249261313); /* 8 */
  FF ( a, b, c, d, getw(in,  8), S11, 1770035416); /* 9 */
  FF ( d, a, b, c, getw(in,  9), S12, 2336552879); /* 10 */
  FF ( c, d, a, b, getw(in, 10), S13, 4294925233); /* 11 */
  FF ( b, c, d, a, getw(in, 11), S14, 2304563134); /* 12 */
  FF ( a, b, c, d, getw(in, 12), S11, 1804603682); /* 13 */
  FF ( d, a, b, c, getw(in, 13), S12, 4254626195); /* 14 */
  FF ( c, d, a, b, getw(in, 14), S13, 2792965006); /* 15 */
  FF ( b, c, d, a, getw(in, 15), S14, 1236535329); /* 16 */
 
  /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG ( a, b, c, d, getw(in,  1), S21, 4129170786); /* 17 */
  GG ( d, a, b, c, getw(in,  6), S22, 3225465664); /* 18 */
  GG ( c, d, a, b, getw(in, 11), S23,  643717713); /* 19 */
  GG ( b, c, d, a, getw(in,  0), S24, 3921069994); /* 20 */
  GG ( a, b, c, d, getw(in,  5), S21, 3593408605); /* 21 */
  GG ( d, a, b, c, getw(in, 10), S22,   38016083); /* 22 */
  GG ( c, d, a, b, getw(in, 15), S23, 3634488961); /* 23 */
  GG ( b, c, d, a, getw(in,  4), S24, 3889429448); /* 24 */
  GG ( a, b, c, d, getw(in,  9), S21,  568446438); /* 25 */
  GG ( d, a, b, c, getw(in, 14), S22, 3275163606); /* 26 */
  GG ( c, d, a, b, getw(in,  3), S23, 4107603335); /* 27 */
  GG ( b, c, d, a, getw(in,  8), S24, 1163531501); /* 28 */
  GG ( a, b, c, d, getw(in, 13), S21, 2850285829); /* 29 */
  GG ( d, a, b, c, getw(in,  2), S22, 4243563512); /* 30 */
  GG ( c, d, a, b, getw(in,  7), S23, 1735328473); /* 31 */
  GG ( b, c, d, a, getw(in, 12), S24, 2368359562); /* 32 */

  /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH ( a, b, c, d, getw(in,  5), S31, 4294588738); /* 33 */
  HH ( d, a, b, c, getw(in,  8), S32, 2272392833); /* 34 */
  HH ( c, d, a, b, getw(in, 11), S33, 1839030562); /* 35 */
  HH ( b, c, d, a, getw(in, 14), S34, 4259657740); /* 36 */
  HH ( a, b, c, d, getw(in,  1), S31, 2763975236); /* 37 */
  HH ( d, a, b, c, getw(in,  4), S32, 1272893353); /* 38 */
  HH ( c, d, a, b, getw(in,  7), S33, 4139469664); /* 39 */
  HH ( b, c, d, a, getw(in, 10), S34, 3200236656); /* 40 */
  HH ( a, b, c, d, getw(in, 13), S31,  681279174); /* 41 */
  HH ( d, a, b, c, getw(in,  0), S32, 3936430074); /* 42 */
  HH ( c, d, a, b, getw(in,  3), S33, 3572445317); /* 43 */
  HH ( b, c, d, a, getw(in,  6), S34,   76029189); /* 44 */
  HH ( a, b, c, d, getw(in,  9), S31, 3654602809); /* 45 */
  HH ( d, a, b, c, getw(in, 12), S32, 3873151461); /* 46 */
  HH ( c, d, a, b, getw(in, 15), S33,  530742520); /* 47 */
  HH ( b, c, d, a, getw(in,  2), S34, 3299628645); /* 48 */

  /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II ( a, b, c, d, getw(in,  0), S41, 4096336452); /* 49 */
  II ( d, a, b, c, getw(in,  7), S42, 1126891415); /* 50 */
  II ( c, d, a, b, getw(in, 14), S43, 2878612391); /* 51 */
  II ( b, c, d, a, getw(in,  5), S44, 4237533241); /* 52 */
  II ( a, b, c, d, getw(in, 12), S41, 1700485571); /* 53 */
  II ( d, a, b, c, getw(in,  3), S42, 2399980690); /* 54 */
  II ( c, d, a, b, getw(in, 10), S43, 4293915773); /* 55 */
  II ( b, c, d, a, getw(in,  1), S44, 2240044497); /* 56 */
  II ( a, b, c, d, getw(in,  8), S41, 1873313359); /* 57 */
  II ( d, a, b, c, getw(in, 15), S42, 4264355552); /* 58 */
  II ( c, d, a, b, getw(in,  6), S43, 2734768916); /* 59 */
  II ( b, c, d, a, getw(in, 13), S44, 1309151649); /* 60 */
  II ( a, b, c, d, getw(in,  4), S41, 4149444226); /* 61 */
  II ( d, a, b, c, getw(in, 11), S42, 3174756917); /* 62 */
  II ( c, d, a, b, getw(in,  2), S43,  718787259); /* 63 */
  II ( b, c, d, a, getw(in,  9), S44, 3951481745); /* 64 */

	a += a0;
	b += b0;
	c += c0;
	d += d0;

}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////


__host__ __device__ void md5_pad(char *paddedWord, char *gpuWord, uint len)
{
	uint i = 0;

	for (; i < len; i++)
		paddedWord[i] = gpuWord[i];
	paddedWord[i] = 0x80;

	i++;
	for (; i < 64; i++)
		paddedWord[i] = 0x0u;
	((uint *)paddedWord)[14] = len * 8; // bit length
}


// The kernel (this is the entrypoint of GPU code)
// Loads the 8-byte word to be hashed from global to shared memory
// and calls the calculation routine
__global__ void md5_calc_l0(char *gpuWords, uint *gpuHashes, int activeThreads)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= activeThreads) { return; }

	__shared__ uint memory[THREADS_PER_BLOCK * 16];

	// load the dictionary word for this thread
	uint *iPaddedWord = &memory[0] + threadIdx.x * 16;
	md5_pad ((char *)iPaddedWord, &gpuWords[MAX_MSG_LEN * idx], MAX_MSG_LEN);

	// compute MD5 hash
	uint a, b, c, d;

	md5_cuda (iPaddedWord, a, b, c, d);

	// return the hash
	gpuHashes[4 * idx + 0] = a;
	gpuHashes[4 * idx + 1] = b;
	gpuHashes[4 * idx + 2] = c;
	gpuHashes[4 * idx + 3] = d; 
}

__global__ void md5_calc_l1(char *gpuWords, uint *gpuHashes, int activeThreads)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= activeThreads) { return; }

	__shared__ uint memory[THREADS_PER_BLOCK * 17];

	// load the dictionary word for this thread
	uint *iPaddedWord = &memory[0] + threadIdx.x * 17;
	md5_pad ((char *)iPaddedWord, &gpuWords[MAX_MSG_LEN * idx], MAX_MSG_LEN);

	// compute MD5 hash
	uint a, b, c, d;

	md5_cuda (iPaddedWord, a, b, c, d);

	// return the hash
	gpuHashes[4 * idx + 0] = a;
	gpuHashes[4 * idx + 1] = b;
	gpuHashes[4 * idx + 2] = c;
	gpuHashes[4 * idx + 3] = d; 
}

__global__ void md5_calc_l2(char *gpuWords, uint *gpuHashes, int activeThreads)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= activeThreads) { return; }

	__shared__ uint memory[THREADS_PER_BLOCK * 17];
	__shared__ char ibuf[THREADS_PER_BLOCK * MAX_MSG_LEN];
	__shared__ uint obuf[THREADS_PER_BLOCK * 4];

	uint chIdx = MAX_MSG_LEN * blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = 0; i < MAX_MSG_LEN; i++)
	{
		ibuf[threadIdx.x + i * THREADS_PER_BLOCK] = gpuWords[chIdx + i * THREADS_PER_BLOCK];
	}

	__syncthreads (); 

	// load the dictionary word for this thread
	uint *iPaddedWord = &memory[0] + threadIdx.x * 17;
	md5_pad ((char *)iPaddedWord, &ibuf[threadIdx.x * MAX_MSG_LEN], MAX_MSG_LEN);

	// compute MD5 hash

	md5_cuda (iPaddedWord, obuf[4 * threadIdx.x], obuf[4 * threadIdx.x + 1], obuf[4 * threadIdx.x + 2], obuf[4 * threadIdx.x + 3]);

	__syncthreads (); 

	uint iIdx = 4 * blockIdx.x * blockDim.x + threadIdx.x;

	gpuHashes[iIdx + 0 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 0 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 1 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 1 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 2 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 2 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 3 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 3 * THREADS_PER_BLOCK]; 
}

__global__ void md5_calc_l3(char *gpuWords, uint *gpuHashes, int activeThreads)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= activeThreads) { return; }

	__shared__ uint memory[THREADS_PER_BLOCK * 17];
	__shared__ char ibuf[THREADS_PER_BLOCK * (MAX_MSG_LEN + 1)];
	__shared__ uint obuf[THREADS_PER_BLOCK * (4 + 1)];

	uint chIdx = (MAX_MSG_LEN + 1) * blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = 0; i < MAX_MSG_LEN + 1; i++)
	{
		ibuf[threadIdx.x + i * THREADS_PER_BLOCK] = gpuWords[chIdx + i * THREADS_PER_BLOCK];
	}

	__syncthreads (); 

	// load the dictionary word for this thread
	uint *iPaddedWord = &memory[0] + threadIdx.x * 17;
	md5_pad ((char *)iPaddedWord, &ibuf[threadIdx.x * (MAX_MSG_LEN + 1)], MAX_MSG_LEN);

	// compute MD5 hash

	md5_cuda (iPaddedWord, obuf[5 * threadIdx.x], obuf[5 * threadIdx.x + 1], obuf[5 * threadIdx.x + 2], obuf[5 * threadIdx.x + 3]);

	__syncthreads (); 

	uint iIdx = 5 * blockIdx.x * blockDim.x + threadIdx.x;

	gpuHashes[iIdx + 0 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 0 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 1 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 1 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 2 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 2 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 3 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 3 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 4 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 4 * THREADS_PER_BLOCK];
}

__global__ void md5_calc_l4(char *gpuWords, uint *gpuHashes, int activeThreads)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= activeThreads) { return; }

	__shared__ uint memory[THREADS_PER_BLOCK * 17];
	__shared__ char ibuf[THREADS_PER_BLOCK * (MAX_MSG_LEN + 4)];
	__shared__ uint obuf[THREADS_PER_BLOCK * (4 + 1)];

	uint chIdx = ((MAX_MSG_LEN + 4) / 4) * blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = 0; i < (MAX_MSG_LEN + 4) / 4; i++)
	{
		((uint *)ibuf)[threadIdx.x + i * THREADS_PER_BLOCK] = ((uint *)gpuWords)[chIdx + i * THREADS_PER_BLOCK];
	}

	__syncthreads (); 

	// load the dictionary word for this thread
	uint *iPaddedWord = &memory[0] + threadIdx.x * 17;
	md5_pad ((char *)iPaddedWord, &ibuf[threadIdx.x * (MAX_MSG_LEN + 4)], MAX_MSG_LEN);

	// compute MD5 hash

	md5_cuda (iPaddedWord, obuf[5 * threadIdx.x], obuf[5 * threadIdx.x + 1], obuf[5 * threadIdx.x + 2], obuf[5 * threadIdx.x + 3]);

	__syncthreads (); 

	uint iIdx = 5 * blockIdx.x * blockDim.x + threadIdx.x;

	gpuHashes[iIdx + 0 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 0 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 1 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 1 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 2 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 2 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 3 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 3 * THREADS_PER_BLOCK];
	gpuHashes[iIdx + 4 * THREADS_PER_BLOCK] = obuf[threadIdx.x + 4 * THREADS_PER_BLOCK];
}

void (*md5_funcs[5]) (char *gpuWords, uint *gpuHashes, int activeThreads);

void setup_md5_funcs ()
{
	md5_funcs[0] = md5_calc_l0;
	md5_funcs[1] = md5_calc_l1;
	md5_funcs[2] = md5_calc_l2;
	md5_funcs[3] = md5_calc_l3;
	md5_funcs[4] = md5_calc_l4;
}


// A helper to export the kernel call to C++ code not compiled with nvcc
double gpu_execute_kernel(char *gpuWords, uint *gpuHashes, int activeThreads, int level)
{
	dim3 grid, block;

	block.x = THREADS_PER_BLOCK, block.y = 1, block.z = 1;
	grid.x = (activeThreads + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK, grid.y = 1, grid.z = 1;

	cudaEvent_t start, stop;
	cudaEventCreate (&start), cudaEventCreate (&stop);
	cudaEventRecord (start, 0);

	md5_funcs[level]<<<grid, block>>>(gpuWords, gpuHashes, activeThreads);

	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	float elapsedTime;
	cudaEventElapsedTime (&elapsedTime, start, stop);
	cudaEventDestroy (start), cudaEventDestroy (stop);

	return elapsedTime;
}

