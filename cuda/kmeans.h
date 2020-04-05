#ifndef KMEANS_H
#define KMEANS_H

namespace kmeans
{
void setsMean(int setid, int *sets, float x[][5], float *mu, const int nObs, const int nFeatures, int *counts, float *sums, int rank);
void updateSets(int *sets, int nSets, int *counts, float *sums, float x[][5], float *mu, const int nObs, const int nFeatures, const int rank, const int npp);
int minIdx(float *arr, int n);
} // namespace kmeans

#endif