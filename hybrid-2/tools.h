#ifndef TOOLS_H
#define TOOLS_H

#include <string>
using namespace std;

namespace tools
{
void readData(string fileName, float *x, const int nFeatures);
void forgy(int nSets, int nFeatures, float *mu, float *x, int nObs);
bool inArray(int *arr, int item, int n);
bool arrayCompare(const int nObs, bool *h_converge);
double CLOCK();

template <class T>
void initArray(const int n, T *arr)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = 0;
    }
}

void saveData(const string saveName, int *sets, int nObs);
void saveStats(double timeStats[6], string fileName);
} // namespace tools

#endif