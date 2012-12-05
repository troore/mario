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
// Convert an array of null-terminated strings to an array of 8-byte words, 
// MD5 padding is pedding to be done.
//
void md5_prep_array(std::valarray<char> &unPaddedWords, const std::vector<std::string> &words, unsigned char *words_len)
{
	uint word_len_sum = 0;

	for (uint i = 0; i < words.size (); i++)
	{
		words_len[i] = strlen (words[i].c_str ()) && 0xff;
		word_len_sum += words_len[i];
	}
	unPaddedWords.resize(word_len_sum);
	unPaddedWords = 0;

	uint local_len_sum = 0;

	for(uint i = 0; i != words.size(); i++)
	{
		char *w = &unPaddedWords[local_len_sum];
		strncpy(w, words[i].c_str(), words_len[i]);
		local_len_sum += words_len[i];
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

	unsigned char *words_len = new unsigned char(ptext.size ());

	md5_prep_array(unPaddedWords, ptext, words_len);

	char *gpuWords;
	uint *gpuHashes = NULL;
	unsigned char *gwords_len;

	double gpuTime = 0.;

	int dynShmemPerThread = 64;	// built in the algorithm

	uint n = ptext.size(), tpb;	// n is number of message words, and tpb is number of threads per block

	int gridDim[3];

	// load the MD5 constant arrays into GPU constant memory
	init_constants();

	hashes.resize(n);

	/**
	 * 8 is the max length of a single message word (00000000~99999999),
	 * 1 is overhead for one element in message length vector,
	 * and as for 16, everybody knows.
	 */
	if (/*1 ||*/ (n * (1 + 8 + 16) < GLOBAL_MEMORY_CAPACITY))
	{
		printf ("Global Memory is still enough!\n");

		// Upload the dictionary onto the GPU device
		cudaMalloc((void **)&gpuWords, unPaddedWords.size());
		cudaMemcpy(gpuWords, &unPaddedWords[0], unPaddedWords.size(), cudaMemcpyHostToDevice);

		// allocate GPU memory for computed hashes
		cudaMalloc((void **)&gpuHashes, n * 4 * sizeof(uint));

		cudaMalloc ((void **)&gwords_len, n);
		cudaMemcpy(gwords_len, words_len, n, cudaMemcpyHostToDevice);

		tpb = 10;
		gridDim[0] = (n + tpb - 1) / tpb, gridDim[1] = 1, gridDim[2] = 1;

		// Call the kernel niters times and calculate the average running time
		for (int k = 0; k != niters; k++)
		{
			gpuTime += gpu_execute_kernel(gridDim[0], gridDim[1], tpb, tpb * dynShmemPerThread, n, gpuWords, gpuHashes, gwords_len);
		}
		gpuTime /= niters;
		// Download the computed hashes
		cudaMemcpy(&hashes.front(), gpuHashes, n * 4 * sizeof(uint), cudaMemcpyDeviceToHost);
	} 
	else
	{
#define CHUNK_NUM 100
		printf ("Global Memory is limited!\n");

		double localTime = 0.;
		uint nChunkSize = (n + CHUNK_NUM - 1) / CHUNK_NUM;

		tpb = 100;
		gridDim[0] = (nChunkSize + tpb - 1) / tpb, gridDim[1] = 1, gridDim[2] = 1;

		cudaMalloc ((void **)&gpuHashes, nChunkSize * 4 * sizeof (uint));
		for (uint i = 0; i < CHUNK_NUM; i++)
		{
			unsigned char *local_words_len = new unsigned char(nChunkSize);
			unsigned char *glocal_words_len;
			cudaMalloc ((void **)glocal_words_len, nChunkSize);
			uint local_words_size = 0;
			for (uint j = 0; j < nChunkSize; j++)
			{
				local_words_len[j] = words_len[i * nChunkSize + j];
				local_words_size = local_words_len[j];
			}
			cudaMemcpy (glocal_words_len, local_words_len, nChunkSize, cudaMemcpyHostToDevice);

			cudaMalloc ((void **)&gpuWords, local_words_size);

			uint pos = 0;
			for (uint j = 0; j < i * nChunkSize; j++)
				pos += words_len[j];
			cudaMemcpy (gpuWords, &unPaddedWords[pos], local_words_size, cudaMemcpyHostToDevice);

			localTime = 0.;
			for (int k = 0; k != niters; k++)
			{
				localTime += gpu_execute_kernel(gridDim[0], gridDim[1], tpb, tpb * dynShmemPerThread, nChunkSize, gpuWords, gpuHashes, glocal_words_len);
			}
			localTime /= niters;

			gpuTime += localTime;

			cudaMemcpy((uint *)(&hashes.front()) + i * nChunkSize * 4, gpuHashes, nChunkSize * 4 * sizeof(uint), cudaMemcpyDeviceToHost);

			cudaFree (gpuWords);
			cudaFree (glocal_words_len);
			delete[] local_words_len;
		}
#undef CHUNK_NUM
	}

	// Shutdown
	cudaFree(gpuHashes);
	cudaFree(gwords_len);

	delete[] words_len;

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

	unsigned char *words_len = new unsigned char(ptext.size ());

	md5_prep_array(unPaddedWords, ptext, words_len);
	hashes.resize(ptext.size());

	char *cpuWords;
	uint *cpuHashes;
	double cpuTime = 0.;

	cpuWords = (char *)&unPaddedWords[0];
	cpuHashes = (uint *)&hashes[0];

	cpuTime = cpu_execute_kernel (cpuWords, cpuHashes, hashes.size (), words_len);

	delete[] words_len;

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
		if(memcmp(hashes1[i].ui, hashes1[i].ui, 16) == 0)
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
	// Load plaintext dictionary
	std::vector<std::string> ptext;
	std::cerr << "Loading words from stdin ...\n";
	std::string word;
	while(std::cin >> word)
	{
		if (word.size () > MAX_MSG_LEN)
		{
			std::cerr << "Message length shouldn't exceed " << MAX_MSG_LEN << std::endl;
			std::cerr << word << " will be ignored...\n";
			continue;
		}
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
