#include <fstream>
#include <iostream>
#include <limits>
#include "readData.h"
#include <sstream>
#include <sys/stat.h>

using namespace std;
using namespace dataRead;

/**
 * @brief Read the CSV file, getting data
 * 
 * @param fileName      CSV filename
 * @param x             The data from the CSV file
 * @param labels        Class label for the data (as ints, ie, 0, 1, 2)
 * @param nFeatures     The number of features that compose an observation
 */
void dataRead::readData(string fileName, float **x, int *labels, const int nFeatures)
{
    string path = "../data_generation/" + fileName;
    cout << path << endl;
    ifstream dataFile(path.c_str()); // open the data file
    string line, value;              // lines and values from file
    char delim = ',';                // delimiter used in data
    int dataItem = 0;                // the current data item
    float x1, x2;
    int id;

    if (!dataFile.is_open())
        cout << "ERROR: " << fileName << " failed to open." << endl;
    else
    {
        // getline(dataFile, line)
        while (getline(dataFile, line)) // loop through lines of the data file
        {
            // x[dataItem][0] = x1;
            // x[dataItem][1] = x2;
            // labels[dataItem] = id;
            // can also use (dataFile >> x1 >> delim >> x2 >> delim >> id) && (delim == ',') in while condition
            // cout << line << endl;
            stringstream s(line); // convert the line to a string stream
            int i = 0;
            while (getline(s, value, delim))
            {
                if (i >= nFeatures)
                {
                    labels[dataItem] = stoi(value);
                    continue;
                }
                x[dataItem][i] = stof(value);
                i++;
            }

            if (dataFile.eof())
                break;

            dataItem++;
        }
        cout << "Data points read: " << dataItem << endl;
    }

    dataFile.close(); // close the file
}

/**
 * @brief Print a sample of the read data
 * 
 * @param sampleSize    The sample size of the data to read
 * @param x             The data
 * @param labels        Class label for the data
 * @param nFeatures     The number of features that compose an observation
 */
void dataRead::printSample(int sampleSize, float **x, int *labels, int nFeatures)
{
    cout << "Data sample: first " << sampleSize << " points" << endl;
    for (int i = 0; i < sampleSize; i++)
    {
        for (int j = 0; j < nFeatures; j++)
        {
            cout << x[i][j] << ", ";
        }
        cout << labels[i] << endl;
    }
}

/**
 * @brief Save the (converged) data labels in an output CSV file
 * 
 * @param saveName  The name of the output file
 * @param sets      The converged labels
 * @param nObs      The number of observations
 */
void dataRead::saveData(const string saveName, int *sets, int nObs)
{
    struct stat buffer;
    if (stat(saveName.c_str(), &buffer) == 0) // check for file accessibility
        cout << "WARNING: output file may be overwritten!" << endl;
    
    ofstream outputData(saveName); // save the file in the current working directory

    for (int i = 0; i < nObs; i++)
    {
        outputData << sets[i] << "\n";
    }

    cout << "Labels saved as \"" << saveName << "\"" << endl;
    outputData.close();
}

void dataRead::saveStats(double timeStats[5], string fileName)
{
    typedef numeric_limits<double> dbl;

    ofstream timeFile(fileName, ofstream::out | ofstream::app);

    timeFile.precision(dbl::max_digits10);
    for (int i = 0; i < 4; i++)
        timeFile << timeStats[i] << ",";
    timeFile << timeStats[4] << "\n";

    timeFile.close();
}