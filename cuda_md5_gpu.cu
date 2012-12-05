// CUDA MD5 hash calculation implementation.

#define RSA_KERNEL md5_v2

#include <stdio.h>

typedef unsigned int uint;

//
// On-device variable declarations
//

extern __shared__ uint memory[];	// on-chip shared memory
__constant__ uint k[64], rconst[16];	// constants (in fast on-chip constant cache)
__constant__ uint target[4];		// target hash, if searching for hash matches

//
// MD5 magic numbers. These will be loaded into on-device "constant" memory
//
static const uint k_cpu[64] =
{
	0xd76aa478, 	0xe8c7b756,	0x242070db,	0xc1bdceee,
	0xf57c0faf,	0x4787c62a, 	0xa8304613,	0xfd469501,
	0x698098d8,	0x8b44f7af,	0xffff5bb1,	0x895cd7be,
	0x6b901122, 	0xfd987193, 	0xa679438e,	0x49b40821,

	0xf61e2562,	0xc040b340, 	0x265e5a51, 	0xe9b6c7aa,
	0xd62f105d,	0x2441453,	0xd8a1e681,	0xe7d3fbc8,
	0x21e1cde6,	0xc33707d6, 	0xf4d50d87, 	0x455a14ed,
	0xa9e3e905,	0xfcefa3f8, 	0x676f02d9, 	0x8d2a4c8a,

	0xfffa3942,	0x8771f681, 	0x6d9d6122, 	0xfde5380c,
	0xa4beea44, 	0x4bdecfa9, 	0xf6bb4b60, 	0xbebfbc70,
	0x289b7ec6, 	0xeaa127fa, 	0xd4ef3085,	0x4881d05,
	0xd9d4d039, 	0xe6db99e5, 	0x1fa27cf8, 	0xc4ac5665,

	0xf4292244, 	0x432aff97, 	0xab9423a7, 	0xfc93a039,
	0x655b59c3, 	0x8f0ccc92, 	0xffeff47d, 	0x85845dd1,
	0x6fa87e4f, 	0xfe2ce6e0, 	0xa3014314, 	0x4e0811a1,
	0xf7537e82, 	0xbd3af235, 	0x2ad7d2bb, 	0xeb86d391,
};

static const uint rconst_cpu[16] =
{
	7, 12, 17, 22,   5,  9, 14, 20,   4, 11, 16, 23,   6, 10, 15, 21
};

void init_constants()
{
	cudaMemcpyToSymbol(k, k_cpu, sizeof(k));
	cudaMemcpyToSymbol(rconst, rconst_cpu, sizeof(rconst));
}

//
// MD5 routines (straight from Wikipedia's MD5 pseudocode description)
//

__device__ inline uint leftrotate (uint x, uint c)
{
	return (x << c) | (x >> (32-c));
}

__device__ inline uint r(const uint i)
{
	return rconst[(i / 16) * 4 + i % 4];
}

__device__ inline uint &getw(uint *w, const int i)
{
	return w[(i+threadIdx.x) % 16];
}

__device__ inline uint getw(const uint *w, const int i)	// const- version
{
//	return w[(i+threadIdx.x) % 16];
	return w[i];
}


__device__ inline uint getk(const int i)
{
	return k[i];	// Note: this is as fast as possible (measured)
}

__device__ void step(const uint i, const uint f, const uint g, uint &a, uint &b, uint &c, uint &d, const uint *w)
{
	uint temp = d;
	d = c;
	c = b;
	b = b + leftrotate((a + f + getk(i) + getw(w, g)), r(i));
	a = temp;
}

__device__ void inline md5(const uint *w, uint &a, uint &b, uint &c, uint &d)
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

	uint f, g, i = 0;
	for(; i != 16; i++)
	{
		f = (b & c) | ((~b) & d);
		g = i;
		step(i, f, g, a, b, c, d, w);
	}

	for(; i != 32; i++)
	{
		f = (d & b) | ((~d) & c);
		g = (5*i + 1) % 16;
		step(i, f, g, a, b, c, d, w);
	}

	for(; i != 48; i++)
	{
		f = b ^ c ^ d;
		g = (3*i + 5) % 16;
		step(i, f, g, a, b, c, d, w);
	}

	for(; i != 64; i++)
	{
		f = c ^ (b | (~d));
		g = (7*i) % 16;
		step(i, f, g, a, b, c, d, w);
	}

	a += a0;
	b += b0;
	c += c0;
	d += d0;
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
void inline __device__ md5_v2(const uint *in, uint &a, uint &b, uint &c, uint &d)
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
__global__ void md5_calc(char *gpuWords, uint *gpuHashes, int realthreads, uint maxWordLen)
{
	uint idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x; // assuming blockDim.y = 1 and threadIdx.y = 0, always
	if (idx >= realthreads) { return; }

	// load the dictionary word for this thread
	uint *iPaddedWord = &memory[0] + threadIdx.x * 16;
	md5_pad ((char *)iPaddedWord, &gpuWords[maxWordLen * idx], maxWordLen);

	// compute MD5 hash
	uint a, b, c, d;

	RSA_KERNEL(iPaddedWord, a, b, c, d);

	// return the hash
	gpuHashes[4 * idx + 0] = a;
	gpuHashes[4 * idx + 1] = b;
	gpuHashes[4 * idx + 2] = c;
	gpuHashes[4 * idx + 3] = d;
}

// A helper to export the kernel call to C++ code not compiled with nvcc
double gpu_execute_kernel(int blocks_x, int blocks_y, int threads_per_block, int shared_mem_required, int realthreads, char *gpuWords, uint *gpuHashes, uint max_word_len)
{
	dim3 grid;
	grid.x = blocks_x; grid.y = blocks_y, grid.z = 1;

	cudaEvent_t start, stop;
	cudaEventCreate (&start), cudaEventCreate (&stop);
	cudaEventRecord (start, 0);

	md5_calc<<<grid, threads_per_block, shared_mem_required>>>(gpuWords, gpuHashes, realthreads, max_word_len);

	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	float elapsedTime;
	cudaEventElapsedTime (&elapsedTime, start, stop);
	cudaEventDestroy (start), cudaEventDestroy (stop);

	return elapsedTime;
}

