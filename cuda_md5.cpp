#include <string.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
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
// Convert an array of null-terminated strings to an array of 8-byte words, 
// MD5 padding is pedding to be done.
//
void md5_prep_array(std::valarray<char> &unPaddedWords, const std::vector<std::string> &words/*, uint &max_word_len*/)
{
//	max_word_len = strlen (words[0].c_str ());

	unPaddedWords.resize(MAX_MSG_LEN * words.size());
	unPaddedWords = 0;

	for(uint i = 0; i != words.size(); i++)
	{
		char *w = &unPaddedWords[i * MAX_MSG_LEN];
		strncpy(w, words[i].c_str(), MAX_MSG_LEN);
//		md5_prep(w);
	}
}

void print_md5(uint *hash, bool crlf)
{
	for(int i = 0; i != 16; i++) { printf("%02x", (uint)(((unsigned char *)hash)[i])); }
	if(crlf) printf("\n");
}



//
// GPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
int cuda_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext)
{

	std::valarray<char> unPaddedWords;

	md5_prep_array(unPaddedWords, ptext);

	char *gpuWords;
	uint *gpuHashes = NULL;

	double gpuTime = 0.;

//	int dynShmemPerThread = 64;	// built in the algorithm

	uint n = ptext.size();	// n is number of message words

//	int gridDim[3];

	// load the MD5 constant arrays into GPU constant memory
	init_constants();

	hashes.resize(n);

	/**
	 * 8 is the max length of a single message word (00000000~99999999),
	 * and as for 16, everybody knows.
	 */
	if (/*1 || */(n * (8 + 16) < GLOBAL_MEMORY_CAPACITY))
	{
		printf ("Calm down, Global Memory is still sufficient.\n");

		// Upload the dictionary onto the GPU device
		cudaMalloc((void **)&gpuWords, unPaddedWords.size());
		cudaMemcpy(gpuWords, &unPaddedWords[0], unPaddedWords.size(), cudaMemcpyHostToDevice);

		// allocate GPU memory for computed hashes
		cudaMalloc((void **)&gpuHashes, n * 4 * sizeof(uint));

//		tpb = THREADS_PER_BLOCK;
//		gridDim[0] = (n + tpb - 1) / tpb, gridDim[1] = 1, gridDim[2] = 1;

		// Call the kernel niters times and calculate the average running time
		for (int k = 0; k != niters; k++)
		{
			gpuTime += gpu_execute_kernel(/*gridDim[0], gridDim[1], tpb, */gpuWords, gpuHashes, n);
		}
		gpuTime /= niters;
		// Download the computed hashes
		cudaMemcpy(&hashes.front(), gpuHashes, n * 4 * sizeof(uint), cudaMemcpyDeviceToHost);
	} 
	else
	{
#define CHUNK_NUM 10
		printf ("Global Memory is limited!\n");

		double localTime = 0.;
//		uint upChunkSize = (unPaddedWords.size() + CHUNK_NUM - 1) / CHUNK_NUM;
		uint nChunkSize = (n + CHUNK_NUM - 1) / CHUNK_NUM;

//		tpb = THREADS_PER_BLOCK;
//		gridDim[0] = (nChunkSize + tpb - 1) / tpb, gridDim[1] = 1, gridDim[2] = 1;

		cudaMalloc ((void **)&gpuWords, MAX_MSG_LEN * nChunkSize);
		cudaMalloc ((void **)&gpuHashes, nChunkSize * 4 * sizeof (uint));
		for (uint i = 0; i < CHUNK_NUM; i++)
		{
			cudaMemcpy (gpuWords, &unPaddedWords[i * MAX_MSG_LEN * nChunkSize], MAX_MSG_LEN * nChunkSize, cudaMemcpyHostToDevice);

			localTime = 0.;
			for (int k = 0; k != niters; k++)
			{
				localTime += gpu_execute_kernel(/*gridDim[0], gridDim[1], tpb, */gpuWords, gpuHashes, nChunkSize);
			}
			localTime /= niters;

			gpuTime += localTime;

			cudaMemcpy((uint *)(&hashes.front()) + i * nChunkSize * 4, gpuHashes, nChunkSize * 4 * sizeof(uint), cudaMemcpyDeviceToHost);

		}
#undef CHUNK_NUM
	}

	// Shutdown
	cudaFree(gpuWords);
	cudaFree(gpuHashes);


	std::cerr << "GPU MD5 time : " <<  gpuTime << "ms\n";


	return 0;
}

//
// CPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
void cpu_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext)
{
	std::valarray<char> unPaddedWords;

//	uint max_word_len;

	md5_prep_array(unPaddedWords, ptext/*, max_word_len*/);
	hashes.resize(ptext.size());

	char *cpuWords;
	uint *cpuHashes;
	double cpuTime = 0.;

	cpuWords = (char *)&unPaddedWords[0];
	cpuHashes = (uint *)&hashes[0];

	cpuTime = cpu_execute_kernel (cpuWords, cpuHashes, hashes.size ());

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
	for(uint i = 0; i != hashes1.size(); i++)
	{
		if(memcmp(hashes1[i].ui, hashes2[i].ui, 16) == 0)
		{
		//	printf("OK   ");
		//	print_md5(hashes1[i].ui);
			uint8_t *p;
			p = (uint8_t *)&hashes2[i].ui[0];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, " ");
			p = (uint8_t *)&hashes2[i].ui[1];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, " ");
			p = (uint8_t *)&hashes2[i].ui[2];
			fprintf (o, "%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3]);
			fprintf (o, " ");
			p = (uint8_t *)&hashes2[i].ui[3];
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

/*	std::ifstream infile ("msglist.txt");

	if (!infile)
	{
		printf ("Fail to open file 'msglist' for reading.\n");
		exit (1);
	} */
	while(/*infile*/std::cin >> word)
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
