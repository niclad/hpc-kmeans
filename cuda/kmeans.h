#ifndef KMEANS_H
#define KMEANS_H

namespace kmeans
{
void setsMean(int setid, int *sets, float **x, float *mu, const int nObs, const int nFeatures);
__global__ void updateSets(float *d_x, float *d_mu, float *d_sums, int *d_counts, int nSets, int nFeatures, int nObs);
int minIdx(float *arr, int n);
} // namespace kmeans

#endif