#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

/**
 * @brief Print the device's properties
 * 
 */
extern void dispDevice()
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("GPU: %s\n", props.name);
}

__global__ void test()
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread ID

    if (t_id >= 1)
        return;

    printf("t_id %d working?\n", t_id);
}

extern void testWrapper()
{
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)1000 / blockSize);
    test<<<gridSize, blockSize>>>();
}

__device__ float
dist(float *currObs, float *currMu, int nFeatures)
{
    float distance = 0;
    for (int i = 0; i < nFeatures; i++)
        distance += (currObs[i] - currMu[i]) * (currObs[i] - currMu[i]);

    return distance;
}

/**
 * @brief Determine if the two previous set is equivalent to the current set.
 * (As an aside, I'm not sure if there's a better (or more appropriate) way of doing this)
 * 
 * @param nSets         The number of sets
 * @param prevSets      The previous set assignments
 * @param sets          The current set assignments
 */
bool arrayCompare(const int nObs, bool *h_converge)
{
    bool equal = true; // assume the arrays aren't equal, at first

    for (int i = 0; i < nObs; i++)
    {
        equal = h_converge[i];
        if (!equal) // check if the item has been found to be matching
            break;
    }

    return equal;
}

extern void runWrapper(float *x, float *mu, float *sums, int *counts, int *sets, int nSets, int nFeatures, int nObs, int rank)
{
    float *d_x, *d_mu, *d_sums;
    int *d_sets, *d_counts, *d_prevSets;
    bool *h_converge, *d_converge;

    h_converge = new bool[nObs];

    size_t obsBytes, muBytes, setsBytes, cntBytes, convBytes;
    obsBytes = sizeof(float) * nObs * nFeatures * 4; // get the sizes in bytes of device arrays
    muBytes = sizeof(float) * nFeatures * nSets;
    setsBytes = sizeof(int) * nObs;
    cntBytes = sizeof(int) * nSets;
    convBytes = sizeof(bool) * nObs;

    cudaMalloc(&d_x, obsBytes); // allocate memory on device
    cudaMalloc(&d_mu, muBytes);
    cudaMalloc(&d_sums, muBytes);
    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_counts, cntBytes);
    cudaMalloc(&d_prevSets, setsBytes);
    cudaMalloc(&d_converge, convBytes);

    cudaMemcpy(d_x, x, obsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sets, sets, setsBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    bool convergence = false;
    int maxIter = 10000, currIter = 0;
    while (!convergence && (currIter < maxIter))
    {
        cudaMemset(d_counts, 0, nSets * sizeof(int));
        cudaMemset(d_sums, 0, nFeatures * nSets * sizeof(int));

        copySets<<<gridSize, blockSize>>>(d_sets, d_prevSets, nObs); // dsave previous set assignments
        cudaDeviceSynchronize();
        updateSets<<<gridSize, blockSize>>>(d_x, d_mu, d_sums, d_counts, d_sets, nSets, nFeatures, nObs, rank);
        cudaDeviceSynchronize();

        computeMu<<<1, nSets>>>(d_mu, d_sums, d_counts, nFeatures);

        checkConvergence<<<gridSize, blockSize>>>(d_sets, d_prevSets, d_converge, nObs);
        cudaMemcpy(h_converge, d_converge, convBytes, cudaMemcpyDeviceToHost);

        convergence = arrayCompare(nObs, h_converge);
        currIter++;
    }

    cudaMemcpy(sums, d_sums, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(counts, d_counts, cntBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sets, d_sets, setsBytes, cudaMemcpyDeviceToHost);

    //printf("counts[0]=%d\n", counts[0]);

    cudaFree(d_x);
    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_sets);
    cudaFree(d_counts);
}

__global__ void updateSets(float *d_x, float *d_mu, float *d_sums, int *d_counts, int *d_sets, int nSets, int nFeatures, int nObs, int rank)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    if (t_id >= nObs)
        return;

    float currObs[5];
    float currMu[5];
    int startIdx = (rank * nObs * nFeatures) + (t_id * nFeatures);

    for (int i = 0; i < nFeatures; i++)
    {
        currObs[i] = d_x[startIdx + i];
    }

    // if (t_id == 0)
    // {
    //     printf("npp %d startIdx %d\n", nObs, startIdx);
    //     printf("displaying t_id 0 pt on rank %d.\n", rank);
    //     for (int i = 0; i < 5; i++)
    //     {
    //         printf("%f ", currObs[i]);
    //     }
    //     printf("\n");
    // }

    float bestDist = FLT_MAX; // maximum float
    int bestSet = 0;
    for (int aSet = 0; aSet < nSets; aSet++)
    {
        for (int i = 0; i < nFeatures; i++)
        {
            currMu[i] = d_mu[(aSet * nFeatures) + i];
        }

        float distance = dist(currObs, currMu, nFeatures);
        if (distance < bestDist)
        {
            bestDist = distance;
            bestSet = aSet;
        }
    }
    d_sets[t_id] = bestSet;
    atomicAdd(&d_counts[bestSet], 1);

    for (int i = 0; i < nFeatures; i++)
    {
        atomicAdd(&d_sums[(bestSet * nFeatures) + i], currObs[i]);
    }
}

extern void updateSetsWrapper(float *x, float *mu, float *sums, int *counts, int *sets, int nSets, int nFeatures, int nObs, int rank)
{
    float *d_x, *d_mu, *d_sums;
    int *d_sets, *d_counts;

    size_t obsBytes, muBytes, setsBytes, cntBytes;
    obsBytes = sizeof(float) * nObs * nFeatures * 4; // get the sizes in bytes of device arrays
    muBytes = sizeof(float) * nFeatures * nSets;
    setsBytes = sizeof(int) * nObs;
    cntBytes = sizeof(int) * nSets;

    cudaMalloc(&d_x, obsBytes); // allocate memory on device
    cudaMalloc(&d_mu, muBytes);
    cudaMalloc(&d_sums, muBytes);
    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_counts, cntBytes);

    cudaMemcpy(d_x, x, obsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    updateSets<<<gridSize, blockSize>>>(d_x, d_mu, d_sums, d_counts, d_sets, nSets, nFeatures, nObs, rank);

    cudaMemcpy(sums, d_sums, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(counts, d_counts, cntBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sets, d_sets, setsBytes, cudaMemcpyDeviceToHost);

    //printf("counts[0]=%d\n", counts[0]);

    cudaFree(d_x);
    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_sets);
    cudaFree(d_counts);
}

__global__ void computeMu(float *mu, float *sums, int *counts, int nFeatures)
{
    int set = threadIdx.x;
    int offset = set * nFeatures;
    int i_counts = 1;
    if (counts[set] != 0)
        i_counts = counts[set];
    for (int i = 0; i < nFeatures; i++)
        mu[offset + i] = sums[offset + i] / i_counts;
}

extern void muWrapper(float *mu, float *sums, int *counts, int nFeatures, int nSets)
{
    float *d_mu, *d_sums;
    int *d_counts;

    size_t muBytes, cntBytes;
    muBytes = sizeof(float) * nFeatures * nSets;
    cntBytes = sizeof(int) * nSets;

    cudaMalloc(&d_mu, muBytes);
    cudaMalloc(&d_sums, muBytes);
    cudaMalloc(&d_counts, cntBytes);

    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);

    computeMu<<<1, 3>>>(d_mu, d_sums, d_counts, nFeatures);

    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

__global__ void copySets(int *sets, int *prevSets, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id >= nObs)
        return;

    prevSets[t_id] = sets[t_id];
}

extern void copyWrapper(int *sets, int *prevSets, int nObs)
{
    int *d_sets, *d_prevSets;
    size_t setsBytes = sizeof(int) * nObs;

    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_prevSets, setsBytes);

    cudaMemcpy(d_sets, sets, setsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevSets, prevSets, setsBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    copySets<<<gridSize, blockSize>>>(d_sets, d_prevSets, nObs);

    cudaMemcpy(prevSets, d_prevSets, setsBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sets);
    cudaFree(d_prevSets);
}

__global__ void checkConvergence(int *sets, int *prevSets, bool *converge, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id >= nObs)
        return;

    // printf("%d, %d\n", sets[t_id], prevSets[t_id]);
    converge[t_id] = sets[t_id] == prevSets[t_id];
    //printf("%d, %d\n",t_id, converge[t_id]);
}

extern void convergeWrapper(int *sets, int *prevSets, bool *converge, int nObs)
{
    int *d_sets, *d_prevSets;
    bool *d_converge;
    size_t setsBytes, convBytes;
    setsBytes = sizeof(int) * nObs;
    convBytes = sizeof(bool) * nObs;

    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_prevSets, setsBytes);
    cudaMalloc(&d_converge, convBytes);

    cudaMemcpy(d_sets, sets, setsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevSets, prevSets, setsBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    checkConvergence<<<gridSize, blockSize>>>(d_sets, d_prevSets, d_converge, nObs);

    cudaMemcpy(converge, d_converge, convBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sets);
    cudaFree(d_prevSets);
    cudaFree(d_converge);
}