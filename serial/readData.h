#ifndef READDATA_H
#define READDATA_H

#include <string>
using namespace std;

namespace dataRead
{
void readData(string fileName, float **x, int *labels, const int nFeatures);
void printSample(int sampleSize, float **x, int *labels, int nFeatures);
void saveData(const string saveName, int *sets, int nObs);
} // namespace dataRead

#endif