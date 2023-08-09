#include <iostream>
#include "dpx.cuh"
#include <stdio.h>

void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    // For each GPU device found, print the information (memory, bandwidth etc.)
    // about the device
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

__global__ void smith_waterman_call() {

    int match = 2;
    int mismatch = -1;
    int gap = -1;

    char ref[] = "ACGTAAC";
    char query[] = "ACTATC";

    int rlen = 7;
    int qlen = 6;

    int score_matrix[8][7] = {};

    score_matrix[0][0] = 0;

    for (int i = 1; i < rlen + 1; i++)
    {
        score_matrix[i][0] = score_matrix[i-1][0] + gap;
    }

    for (int i = 1; i < qlen + 1; i++)
    {
        score_matrix[0][i] = score_matrix[0][i-1] + gap;
    }

    for (int i = 1; i < rlen + 1; i++)
    {
        for (int j = 1; j < qlen + 1; j++)
        {
            int up = score_matrix[i-1][j] + gap;
            int left = score_matrix[i][j-1]  + gap;
            int diag = score_matrix[i-1][j-1];
            if (ref[i-1] == query[j-1])
                diag += match;
            else
                diag += mismatch;

            score_matrix[i][j] = __vimax3_s32(diag, left, up); 
        }
    }

    printf("\t");
    for (int j = 0; j < qlen; j++)
    {
        printf("%c\t", query[j]); 
    }
    printf("\n"); 

    for (int i = 1; i < rlen + 1; i++)
    {
        printf("%c\t", ref[i-1]); 
        for (int j = 1; j < qlen + 1; j++)
        {
            printf("%d\t", score_matrix[i][j]); 
        }
        printf("\n"); 
    }

}

void dpx::smith_waterman() {

    int numBlocks = 1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

    smith_waterman_call<<<numBlocks, blockSize>>>();

    cudaDeviceSynchronize();
}