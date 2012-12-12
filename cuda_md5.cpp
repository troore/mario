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
// Convert an array of null-terminated strings to an array of 8-byte words, 
// MD5 padding is pedding to be done.
//
void md5_prep_array(std::valarray<char> &unPaddedWords, const std::vector<std::string> &words)
{
//	max_word_len = strlen (words[0].c_str ());

	unPaddedWords.resize(MAX_MSG_LEN * words.size());
	unPaddedWords = 0;

	for(uint i = 0; i != words.size(); i++)
	{
		char *w = &unPaddedWords[i * MAX_MSG_LEN];
		strncpy(w, words[i].c_str(), MAX_MSG_LEN);
	}
}

void md5_prep_array_v2(std::valarray<char> &unPaddedWords, const std::vector<std::string> &words)
{
	unPaddedWords.resize((MAX_MSG_LEN + 1) * words.size());
	unPaddedWords = 0;

	for(uint i = 0; i != words.size(); i++)
	{
		char *w = &unPaddedWords[i * (MAX_MSG_LEN + 1)];
		strncpy(w, words[i].c_str(), MAX_MSG_LEN);
	}
}

void md5_prep_array_v3(std::valarray<char> &unPaddedWords, const std::vector<std::string> &words)
{
	unPaddedWords.resize((MAX_MSG_LEN + 4) * words.size());
	unPaddedWords = 0;

	for(uint i = 0; i != words.size(); i++)
	{
		char *w = &unPaddedWords[i * (MAX_MSG_LEN + 4)];
		strncpy(w, words[i].c_str(), MAX_MSG_LEN);
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
int cuda_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext, int level)
{

	setup_md5_funcs ();

	std::valarray<char> unPaddedWords;

	md5_prep_array(unPaddedWords, ptext);

	char *gpuWords;
	uint *gpuHashes = NULL;

	double gpuTime = 0.;

	uint n = ptext.size();	// n is number of message words

	hashes.resize(n);

	/**
	 * 8 is the max length of a single message word (00000000~99999999),
	 * and as for 16, everybody knows.
	 */
	if (/*0 &&*/ 1 || (n * (8 + 16) < GLOBAL_MEMORY_CAPACITY))
	{
		printf ("Calm down, Global Memory is still sufficient.\n");

		// Upload the dictionary onto the GPU device
		cudaMalloc((void **)&gpuWords, unPaddedWords.size());
		cudaMemcpy(gpuWords, &unPaddedWords[0], unPaddedWords.size(), cudaMemcpyHostToDevice);

		// allocate GPU memory for computed hashes
		cudaMalloc((void **)&gpuHashes, n * 4 * sizeof(uint));

		// Call the kernel niters times and calculate the average running time
		for (int k = 0; k != niters; k++)
		{
			gpuTime += gpu_execute_kernel(gpuWords, gpuHashes, n, level);
		}
		gpuTime /= niters;
		// Download the computed hashes
		cudaMemcpy(&hashes.front(), gpuHashes, n * 4 * sizeof(uint), cudaMemcpyDeviceToHost);

		// Shutdown
		cudaFree(gpuWords);
		cudaFree(gpuHashes);
	} 
	else
	{
#define CHUNK_NUM 10
		printf ("Global Memory is limited!\n");

		double localTime = 0.;
		uint nChunkSize = (n + CHUNK_NUM - 1) / CHUNK_NUM;

		cudaMalloc ((void **)&gpuWords, MAX_MSG_LEN * nChunkSize);
		cudaMalloc ((void **)&gpuHashes, nChunkSize * 4 * sizeof (uint));
		for (uint i = 0; i < CHUNK_NUM; i++)
		{
			cudaMemcpy (gpuWords, &unPaddedWords[i * MAX_MSG_LEN * nChunkSize], MAX_MSG_LEN * nChunkSize, cudaMemcpyHostToDevice);

			localTime = 0.;
			for (int k = 0; k != niters; k++)
			{
				localTime += gpu_execute_kernel(gpuWords, gpuHashes, nChunkSize, level);
			}
			localTime /= niters;

			gpuTime += localTime;

			cudaMemcpy((char *)(&hashes.front()) + i * nChunkSize * 4 * sizeof (uint), gpuHashes, nChunkSize * 4 * sizeof(uint), cudaMemcpyDeviceToHost);
		}

		cudaFree(gpuWords);
		cudaFree(gpuHashes);
#undef CHUNK_NUM
	}


	std::cerr << "GPU MD5 time : " <<  gpuTime << "ms\n";


	return 0;
}

int cuda_compute_md5s_v2(std::vector<md5hash_v2> &hashes, const std::vector<std::string> &ptext, int level)
{
	setup_md5_funcs ();

	std::cerr << "CUDA md5 v2.\n";

	std::valarray<char> unPaddedWords;

	md5_prep_array_v2(unPaddedWords, ptext);

	char *gpuWords;
	uint *gpuHashes = NULL;

	double gpuTime = 0.;

	uint n = ptext.size();	// n is number of message words

	hashes.resize(n);

	if (/*0 &&*/ 1 || (n * (8 + 1 + 16 + 4) < GLOBAL_MEMORY_CAPACITY))
	{
		printf ("Calm down, Global Memory is still sufficient.\n");

		// Upload the dictionary onto the GPU device
		cudaMalloc((void **)&gpuWords, unPaddedWords.size());
		cudaMemcpy(gpuWords, &unPaddedWords[0], unPaddedWords.size(), cudaMemcpyHostToDevice);

		// allocate GPU memory for computed hashes
		cudaMalloc((void **)&gpuHashes, n * (4 + 1) * sizeof(uint));

		// Call the kernel niters times and calculate the average running time
		for (int k = 0; k != niters; k++)
		{
			gpuTime += gpu_execute_kernel(gpuWords, gpuHashes, n, level);
		}
		gpuTime /= niters;
		// Download the computed hashes
		cudaMemcpy(&hashes.front(), gpuHashes, n * (4 + 1) * sizeof(uint), cudaMemcpyDeviceToHost);

		// Shutdown
		cudaFree(gpuWords);
		cudaFree(gpuHashes);
	} 
	else
	{
#define CHUNK_NUM 10
		printf ("Global Memory is limited!\n");

		double localTime = 0.;
		uint nChunkSize = (n + CHUNK_NUM - 1) / CHUNK_NUM;

		cudaMalloc ((void **)&gpuWords, (MAX_MSG_LEN + 1) * nChunkSize);
		cudaMalloc ((void **)&gpuHashes, nChunkSize * (4 + 1) * sizeof (uint));
		for (uint i = 0; i < CHUNK_NUM; i++)
		{
			cudaMemcpy (gpuWords, &unPaddedWords[i * (MAX_MSG_LEN + 1) * nChunkSize], (MAX_MSG_LEN + 1) * nChunkSize, cudaMemcpyHostToDevice);

			localTime = 0.;
			for (int k = 0; k != niters; k++)
			{
				localTime += gpu_execute_kernel(gpuWords, gpuHashes, nChunkSize, level);
			}
			localTime /= niters;

			gpuTime += localTime;

			cudaMemcpy((char *)(&hashes.front()) + i * nChunkSize * (4 + 1) * sizeof (uint), gpuHashes, nChunkSize * (4 + 1) * sizeof(uint), cudaMemcpyDeviceToHost);
		}

		cudaFree(gpuWords);
		cudaFree(gpuHashes);
#undef CHUNK_NUM
	}

	std::cerr << "GPU MD5 time : " <<  gpuTime << "ms\n";

	return 0;
}

int cuda_compute_md5s_v3(std::vector<md5hash_v2> &hashes, const std::vector<std::string> &ptext, int level)
{
	setup_md5_funcs ();

	std::cerr << "CUDA md5 v3.\n";

	std::valarray<char> unPaddedWords;

	md5_prep_array_v3(unPaddedWords, ptext);

	char *gpuWords;
	uint *gpuHashes = NULL;

	double gpuTime = 0.;

	uint n = ptext.size();	// n is number of message words

	hashes.resize(n);

	if (/*0 &&*/ 1 || (n * (8 + 4 + 16 + 4) < GLOBAL_MEMORY_CAPACITY))
	{
		printf ("Calm down, Global Memory is still sufficient.\n");

		// Upload the dictionary onto the GPU device
		cudaMalloc((void **)&gpuWords, unPaddedWords.size());
		cudaMemcpy(gpuWords, &unPaddedWords[0], unPaddedWords.size(), cudaMemcpyHostToDevice);

		// allocate GPU memory for computed hashes
		cudaMalloc((void **)&gpuHashes, n * (4 + 1) * sizeof(uint));

		// Call the kernel niters times and calculate the average running time
		for (int k = 0; k != niters; k++)
		{
			gpuTime += gpu_execute_kernel(gpuWords, gpuHashes, n, level);
		}
		gpuTime /= niters;
		// Download the computed hashes
		cudaMemcpy(&hashes.front(), gpuHashes, n * (4 + 1) * sizeof(uint), cudaMemcpyDeviceToHost);

		// Shutdown
		cudaFree(gpuWords);
		cudaFree(gpuHashes);
	} 
	else
	{
#define CHUNK_NUM 10
		printf ("Global Memory is limited!\n");

		double localTime = 0.;
		uint nChunkSize = (n + CHUNK_NUM - 1) / CHUNK_NUM;

		cudaMalloc ((void **)&gpuWords, (MAX_MSG_LEN + 4) * nChunkSize);
		cudaMalloc ((void **)&gpuHashes, nChunkSize * (4 + 1) * sizeof (uint));
		for (uint i = 0; i < CHUNK_NUM; i++)
		{
			cudaMemcpy (gpuWords, &unPaddedWords[i * (MAX_MSG_LEN + 4) * nChunkSize], (MAX_MSG_LEN + 4) * nChunkSize, cudaMemcpyHostToDevice);

			localTime = 0.;
			for (int k = 0; k != niters; k++)
			{
				localTime += gpu_execute_kernel(gpuWords, gpuHashes, nChunkSize, level);
			}
			localTime /= niters;

			gpuTime += localTime;

			cudaMemcpy((char *)(&hashes.front()) + i * nChunkSize * (4 + 1) * sizeof (uint), gpuHashes, nChunkSize * (4 + 1) * sizeof(uint), cudaMemcpyDeviceToHost);
		}

		cudaFree(gpuWords);
		cudaFree(gpuHashes);
#undef CHUNK_NUM
	}

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
template <typename elemType>
void compare_hashes(std::vector<elemType> &hashes1, std::vector<md5hash> &hashes2, const std::vector<std::string> &ptext)
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
/*			uint8_t *p;
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
			fprintf (o, "\n"); */
		}
		else
		{
			printf("%-10s ", ptext[i].c_str());
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

void process_args (int argc, char **argv, std::string &infilename, int &optLevel, bool &cpu)
{
	std::string argi;

	if (argc < 2 || argc > 4)
	{
		std::cerr << "Usage: ./cuda_md5 inputfile [-Lx] [-cpu]\n";
		exit (1);
	}
	infilename = argv[1];
	if (argc == 2)
	{
		optLevel = 0;

		return;
	}
	optLevel = argv[2][2] - '0';
	if (argc == 3)
	{
		optLevel = argv[2][2] - '0';

		return;
	}
	if (argc == 4)
	{
		cpu = true;
	}
}


void md5_hash (int &level, std::vector<std::string> &ptext, bool &cpu)
{
	std::vector<md5hash> hashes_gpu;
	std::vector<md5hash_v2> hashes_gpu_v2;

	// Do calculation
	switch (level)
	{
		case 0:
		case 1:
		case 2:
			// Compute hashes
			cuda_compute_md5s(hashes_gpu, ptext, level);
		break;
		case 3:
		cuda_compute_md5s_v2(hashes_gpu_v2, ptext, level);
		break;
		case 4:
		cuda_compute_md5s_v3(hashes_gpu_v2, ptext, level);
		break;
	}

	// Verify the answers
	if (cpu)
	{
		std::vector<md5hash> hashes_cpu;
		cpu_compute_md5s(hashes_cpu, ptext);
		if (level == 0 || level == 1 || level == 2)
			compare_hashes(hashes_gpu, hashes_cpu, ptext);
		if (level == 3 || level == 4)
			compare_hashes(hashes_gpu_v2, hashes_cpu, ptext);
	}
}

int main(int argc, char **argv)
{
	std::string infilename;
	int optLevel;
	bool cpu;

	process_args (argc, argv, infilename, optLevel, cpu);

	std::cout << infilename << "\t" << optLevel << "\t" << cpu << "\n";

	// Load plaintext dictionary
	std::vector<std::string> ptext;
	std::cerr << "Loading words from stdin ...\n";
	std::string word;

	std::ifstream infile (infilename.c_str ());

	if (!infile)
	{
		std::cerr << "Cannot open file " << infilename << " for reading.\n";
		exit (1);
	} 
	while(infile /*std::cin*/ >> word)
	{
		ptext.push_back(word);
	}
	std::cerr << "Loaded " << ptext.size() << " words.\n\n";

	md5_hash (optLevel, ptext, cpu);

	return 0;
}
