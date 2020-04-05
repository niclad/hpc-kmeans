#ifndef TOOLS_H
#define TOOLS_H

namespace tools
{
void randomPartition(int nSets, int *sets, int nObs);
void forgy(int nSets, int nFeatures, float *mu, float **x, int nObs);
bool inArray(int *arr, int item, int n);
bool arrayCompare(const int nObs, int *h_converge);
void arrayCopy(const int nObs, int *prevSets, int *sets);
double CLOCK();
} // namespace tools

#endif