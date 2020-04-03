#ifndef KMEANS_H
#define KMEANS_H

namespace kmeans
{
void setsMean(int setid, int *sets, float **x, float *mu, const int nObs, const int nFeatures);
void updateSets(int *sets, int nSets, float **x, float *mu, const int nObs, const int nFeatures);
int minIdx(float *arr, int n);
} // namespace kmeans

#endif