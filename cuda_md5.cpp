#include <string.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <valarray>
#include <stdint.h>

#include <cuda_runtime.h>
#include "cuda_md5.h"

int niters = 10;

//
// Shared aux. functions (used both by GPU and CPU setup code)
//
union md5hash
{
	uint ui[4];
	char ch[16];
};

//
// Utils
//

// prepare a 56-byte (maximum) wide md5 message by appending the 64-bit length
// we assume c0 is zero-padded
void md5_prep(char *c0)
{
	uint len = 0;
	char *c = c0;
	while(*c) {len++; c++;}

	c[0] = 0x80;			// bit 1 after the message

	/* This padding step is probably unnecessary, 
	 * since the initial values in the vector 'paddedWords'
	 * defined in cuda_compute_md5s is 0. */
/*	uint padding_len = c0 + 64 - ++c;
	for (uint i = 0; i < padding_len; i++)
	{
		c[i] = 0x0u;
	} */


	((uint*)c0)[14] = len * 8;	// message length in bits
}

void print_md5(uint *hash, bool crlf)
{
	for(int i = 0; i != 16; i++) { printf("%02x", (uint)(((unsigned char *)hash)[i])); }
	if(crlf) printf("\n");
}

//
// Convert an array of null-terminated strings to an array of 64-byte words
// with proper MD5 padding
//
void md5_prep_array(std::valarray<char> &paddedWords, const std::vector<std::string> &words)
{
	paddedWords.resize(64 * words.size());
	paddedWords = 0;

	for(uint i = 0; i != words.size(); i++)
	{
		char *w = &paddedWords[i*64];
		strncpy(w, words[i].c_str(), 56);
		md5_prep(w);
	}
}


//
// GPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
int cuda_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext)
{

	// pad dictionary words to 64 bytes (MD5 block size)
	std::valarray<char> paddedWords;
	md5_prep_array(paddedWords, ptext);

	uint *gpuWords, *gpuHashes = NULL;

	double gpuTime = 0.;

	int dynShmemPerThread = 64;	// built in the algorithm

	uint n = ptext.size(), tpb;	// n is number of message words, and tpb is number of threads per block

	int gridDim[3];

	// load the MD5 constant arrays into GPU constant memory
	init_constants();

	hashes.resize(n);

	/**
	 * 8 is the max length of a single message word (00000000~99999999),
	 * and as for 16, everybody knows.
	 */
#define CHUNK_NUM 100
	if (n * (8 + 16) < GLOBAL_MEMORY_CAPACITY)
	{
		printf ("Global Memory is still enough!\n");

		// Upload the dictionary onto the GPU device
		cudaMalloc((void **)&gpuWords, paddedWords.size());
		cudaMemcpy(gpuWords, &paddedWords[0], paddedWords.size(), cudaMemcpyHostToDevice);

		// allocate GPU memory for computed hashes
		cudaMalloc((void **)&gpuHashes, n * 4 * sizeof(uint));

		tpb = 100;
		gridDim[0] = (n + tpb - 1) / tpb, gridDim[1] = 1, gridDim[2] = 1;

		// Call the kernel niters times and calculate the average running time
		for (int k = 0; k != niters; k++)
		{
			gpuTime += gpu_execute_kernel(gridDim[0], gridDim[1], tpb, tpb * dynShmemPerThread, n, gpuWords, gpuHashes);
		}
		gpuTime /= niters;
		// Download the computed hashes
		cudaMemcpy(&hashes.front(), gpuHashes, n * 4 * sizeof(uint), cudaMemcpyDeviceToHost);
	} 
	else
	{
		printf ("Global Memory is limited!\n");

		double localTime = 0.;
		uint pChunkSize = (paddedWords.size() + CHUNK_NUM - 1) / CHUNK_NUM;
		uint nChunkSize = (n + CHUNK_NUM - 1) / CHUNK_NUM;

		tpb = 100;
		gridDim[0] = (nChunkSize + tpb - 1) / tpb, gridDim[1] = 1, gridDim[2] = 1;

		cudaMalloc ((void **)&gpuWords, pChunkSize);
		cudaMalloc ((void **)&gpuHashes, nChunkSize * 4 * sizeof (uint));
		for (uint i = 0; i < CHUNK_NUM; i++)
		{
			cudaMemcpy (gpuWords, &paddedWords[i * pChunkSize], pChunkSize, cudaMemcpyHostToDevice);

			localTime = 0.;
			for (int k = 0; k != niters; k++)
			{
				localTime += gpu_execute_kernel(gridDim[0], gridDim[1], tpb, tpb * dynShmemPerThread, nChunkSize, gpuWords, gpuHashes);
			}
			localTime /= niters;

			gpuTime += localTime;

			cudaMemcpy((uint *)(&hashes.front()) + i * nChunkSize * 4, gpuHashes, nChunkSize * 4 * sizeof(uint), cudaMemcpyDeviceToHost);

		}
	}

	// Shutdown
	cudaFree(gpuWords);
	cudaFree(gpuHashes);

#undef CHUNK_NUM

	std::cerr << "GPU MD5 time : " <<  gpuTime << "ms\n";


	return 0;
}

//
// CPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
void cpu_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext)
{
	std::valarray<char> paddedWords;

	md5_prep_array(paddedWords, ptext);
	hashes.resize(ptext.size());

	uint *cpuWords, *cpuHashes;
	double cpuTime = 0.;

	cpuWords = (uint *)&paddedWords[0];
	cpuHashes = (uint *)&hashes[0];

	cpuTime = cpu_execute_kernel (cpuWords, cpuHashes, hashes.size ());

/*#define CHUNK_NUM 100

	uint pChunkSize = (paddedWords.size () + CHUNK_NUM - 1) / CHUNK_NUM;
	uint nChunkSize = (hashes.size () + CHUNK_NUM - 1) / CHUNK_NUM;
	for (uint i = 0; i < CHUNK_NUM; i++)
	{
		cpuWords = (uint *)&paddedWords[i * pChunkSize];
		cpuHashes = (uint *)&hashes[i * nChunkSize];
		cpuTime += cpu_execute_kernel (cpuWords, cpuHashes, nChunkSize);
	}

#undef CHUNK_NUM */
	std::cerr << "CPU MD5 time : " <<  cpuTime << "ms\n";
}

//
// Compare and print the MD5 hashes hashes1 and hashes2 of plaintext vector
// ptext. Complain if they don't match.
//
void compare_hashes(std::vector<md5hash> &hashes1, std::vector<md5hash> &hashes2, const std::vector<std::string> &ptext)
{
	FILE *o;
	o = fopen ("dataOut", "w");
	if (NULL == o)
	{
		printf ("FAIL!\n");
		exit (1);
	}

	// Compare & print
	for(uint i=0; i != hashes1.size(); i++)
	{
		if(memcmp(hashes1[i].ui, hashes2[i].ui, 16) == 0)
		{
		//	printf("OK   ");
		//	print_md5(hashes1[i].ui);
			uint8_t *p;
			p = (uint8_t *)&hashes1[i].ui[0];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, " ");
			p = (uint8_t *)&hashes1[i].ui[1];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, " ");
			p = (uint8_t *)&hashes1[i].ui[2];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, " ");
			p = (uint8_t *)&hashes1[i].ui[3];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, "\n");
		}
		else
		{
			printf("%-56s ", ptext[i].c_str());
			printf("ERROR   ");
			print_md5(hashes1[i].ui, false);
			printf(" != ");
			print_md5(hashes2[i].ui);
			std::cerr << "Hash " << i << " didn't match. Test failed. Aborting.\n";
			return;
		} 
	}
	std::cerr << "All hashes match.\n";

	fclose (o);
}

int main(int argc, char **argv)
{
/*	option_reader o;

	bool devQuery = false, benchmark = false;
	std::string target_word;

	o.add("deviceQuery", devQuery, option_reader::flag);
	o.add("benchmark", benchmark, option_reader::flag);
	o.add("search", target_word, option_reader::optparam);
	o.add("benchmark-iters", niters, option_reader::optparam);

	if(!o.process(argc, argv))
	{
		std::cerr << "Usage: " << o.cmdline_usage(argv[0]) << "\n";
		return -1;
	} 

	if(devQuery) { return deviceQuery(); } */


	// Load plaintext dictionary
	std::vector<std::string> ptext;
	std::cerr << "Loading words from stdin ...\n";
	std::string word;
	while(std::cin >> word)
	{
		ptext.push_back(word);
	}
	std::cerr << "Loaded " << ptext.size() << " words.\n\n";

	// Do search/calculation
	std::vector<md5hash> hashes_cpu, hashes_gpu;
	// Compute hashes
	cuda_compute_md5s(hashes_gpu, ptext);
	cpu_compute_md5s(hashes_cpu, ptext);

	// Verify the answers
	compare_hashes(hashes_gpu, hashes_cpu, ptext);

	return 0;
}
