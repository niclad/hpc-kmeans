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

// test kernel
__global__ void test()
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread ID

    if (t_id >= 1)
        return;

    printf("t_id %d working?\n", t_id);
}

// call the test kernel
extern void testWrapper()
{
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)1000 / blockSize);
    test<<<gridSize, blockSize>>>();
}

/**
 * @brief Calculate the distance between two points
 * 
 * @param currObs 
 * @param currMu 
 * @param nFeatures 
 * @return __device__ dist 
 */
__device__ float
dist(float *currObs, float *currMu, int nFeatures)
{
    float distance = 0; // initial distance
    for (int i = 0; i < nFeatures; i++)
        distance += (currObs[i] - currMu[i]) * (currObs[i] - currMu[i]); // cumulatively add the vector compenents

    return distance;
}

/**
 * @brief Update set assignments
 * 
 * @param d_x 
 * @param d_mu 
 * @param d_sums 
 * @param d_counts 
 * @param d_sets 
 * @param nSets 
 * @param nFeatures 
 * @param nObs 
 * @param rank 
 * @return __global__ updateSets 
 */
__global__ void updateSets(float *d_x, float *d_mu, float *d_sums, int *d_counts, int *d_sets, int nSets, int nFeatures, int nObs, int rank)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    if (t_id >= nObs) // end the run for a thread out of range
        return;

    float currObs[5]; // the current observation
    float currMu[5];  // the current set mean
    int startIdx = (rank * nObs * nFeatures) + (t_id * nFeatures); // the ranks starting index

    // get the current observation
    for (int i = 0; i < nFeatures; i++)
    {
        currObs[i] = d_x[startIdx + i];
    }

    // if (t_id == 0) // DEBUGGING
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
    int bestSet = 0; // assume best set is 0

    // check all sets
    for (int aSet = 0; aSet < nSets; aSet++)
    {
        // get the sets mean -- could be done a better way, I'm sure 
        for (int i = 0; i < nFeatures; i++)
        {
            currMu[i] = d_mu[(aSet * nFeatures) + i];
        }

        float distance = dist(currObs, currMu, nFeatures); // get the distance to aSet's mean
        if (distance < bestDist) // update distance if current distance is best
        {
            bestDist = distance;
            bestSet = aSet;
        }
    }
    d_sets[t_id] = bestSet; // assign set
    atomicAdd(&d_counts[bestSet], 1); // add 1 to set counts

    for (int i = 0; i < nFeatures; i++) // sum set observations
    {
        atomicAdd(&d_sums[(bestSet * nFeatures) + i], currObs[i]);
    }
}

/**
 * @brief Update set assignments wrapper -- called by an MPI process
 * 
 * @param x 
 * @param mu 
 * @param sums 
 * @param counts 
 * @param sets 
 * @param nSets 
 * @param nFeatures 
 * @param nObs 
 * @param rank 
 */
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

    // copy data from host to device
    cudaMemcpy(d_x, x, obsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice);

    // SET UP threads
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    updateSets<<<gridSize, blockSize>>>(d_x, d_mu, d_sums, d_counts, d_sets, nSets, nFeatures, nObs, rank);

    // copy data from device to host
    cudaMemcpy(sums, d_sums, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(counts, d_counts, cntBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sets, d_sets, setsBytes, cudaMemcpyDeviceToHost);

    //printf("counts[0]=%d\n", counts[0]);

    // free device memory
    cudaFree(d_x);
    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_sets);
    cudaFree(d_counts);
}

/**
 * @brief Update the mean
 * 
 * @param mu 
 * @param sums 
 * @param counts 
 * @param nFeatures 
 * @return __global__ computeMu 
 */
__global__ void computeMu(float *mu, float *sums, int *counts, int nFeatures)
{
    int set = threadIdx.x; // set is current thread
    int offset = set * nFeatures; // index offset

    int i_counts = 1;
    if (counts[set] != 0) // keep from dividing by 0
        i_counts = counts[set];

    // calculate a sets new mean
    for (int i = 0; i < nFeatures; i++)
        mu[offset + i] = sums[offset + i] / i_counts;
}

/**
 * @brief wrapper for computeMu
 * 
 * @param mu 
 * @param sums 
 * @param counts 
 * @param nFeatures 
 * @param nSets 
 */
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

    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice); // copy data from host to device
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);

    computeMu<<<1, 3>>>(d_mu, d_sums, d_counts, nFeatures);

    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost); // copy data from device to host

    cudaFree(d_mu); // freee device memory
    cudaFree(d_sums);
    cudaFree(d_counts);
}


/**
 * @brief Copy an array
 * 
 * @param sets 
 * @param prevSets 
 * @param nObs 
 * @return __global__ copySets 
 */
__global__ void copySets(int *sets, int *prevSets, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id >= nObs)
        return;

    prevSets[t_id] = sets[t_id];
}

/**
 * @brief copySets wrapper
 * 
 * @param sets 
 * @param prevSets 
 * @param nObs 
 */
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

/**
 * @brief Check for convergence
 * 
 * @param sets 
 * @param prevSets 
 * @param converge 
 * @param nObs 
 * @return __global__ checkConvergence 
 */
__global__ void checkConvergence(int *sets, int *prevSets, bool *converge, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id >= nObs)
        return;

    // printf("%d, %d\n", sets[t_id], prevSets[t_id]);
    converge[t_id] = sets[t_id] == prevSets[t_id];
    //printf("%d, %d\n",t_id, converge[t_id]);
}

/**
 * @brief Check convergence wrapper
 * 
 * @param sets 
 * @param prevSets 
 * @param converge 
 * @param nObs 
 */
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