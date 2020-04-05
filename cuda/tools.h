#ifndef TOOLS_H
#define TOOLS_H

namespace tools
{
void randomPartition(int nSets, int *sets, int nObs);
void forgy(int nSets, int nFeatures, float *mu, float x[][5], int nObs);
bool inArray(int *arr, int item, int n);
char arrayCompare(const int nObs, int *prevSets, int *sets);
void arrayCopy(const int nObs, int *prevSets, int *sets);
double CLOCK();

template <class T>
void initArray(const int n, T *arr)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = 0;
    }
}
} // namespace tools

#endif