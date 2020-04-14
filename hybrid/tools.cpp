#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include "tools.h"


using namespace std;

/**
 * @brief Read data from an input CSV file
 * 
 * @param fileName  The name of the file
 * @param x         Data array for store reads
 * @param nFeatures The number of features in the data
 */
void tools::readData(string fileName, float *x, const int nFeatures)
{
    string path = "../data_generation/" + fileName;
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
                    continue;
                
                x[dataItem] = stof(value);
                dataItem++;
                i++;
                // cout << dataItem << endl;
            }

            if (dataFile.eof())
                break;
        }
        
        cout << "Data points read: " << dataItem / nFeatures << endl;
    }

    dataFile.close(); // close the file
}

/**
 * @brief K-means Forgy initialization
 * 
 * @param nSets     The number of sets (read clusters)
 * @param nFeatures The number of features in the data
 * @param mu        The set means
 * @param x         Data points
 * @param nObs      The number of observations
 */
void tools::forgy(int nSets, int nFeatures, float *mu, float *x, int nObs)
{
    int val = 0;
    int *obsChoice = new int[nSets]; // the observations to choose from 
    for (int i = 0; i < nSets; i++) // initialize the array
        obsChoice[i] = -1;

    random_device seed;
    mt19937 rng(seed());
    uniform_int_distribution<mt19937::result_type> dist(1, nObs);

    for (int i = 0; i < nSets; i++)
    {
        bool inArr = true;
        while (inArr) // while the value picked is in the choices, pick a value
        {
            val = dist(rng);
            inArr = inArray(obsChoice, val, nSets);
        }
        obsChoice[i] = val;
        //cout << val << endl;
    }

    // hmmm...

    for (int i = 0; i < nSets; i++) // assign the starting means
    {
        for (int j = 0; j < nFeatures; j++)
        {
            int offset = nFeatures * i; // calculate the sets mu index offset
            int rowIdx = (obsChoice[i] * nFeatures) - nFeatures; // get the row index from choices
            mu[j + offset] = x[rowIdx + j];
        }
    }

    delete[] obsChoice; // free host memory
}

/**
 * @brief Check for an item in an array
 * 
 * @param arr   The array to check
 * @param item  The item to look for
 * @param n     The number of items to search
 * @return true 
 * @return false 
 */
bool tools::inArray(int *arr, int item, int n)
{
    bool inArr = true; // assume the element is in the array

    for (int i = 0; i < n; i++)
    {
        inArr = arr[i] == item;
        if (inArr) // the arraty is in the list, end the check
            break;
    }

    return inArr;
}

/**
 * @brief Check for array equivalence
 * 
 * @param nObs          The number of elements to compare
 * @param h_converge    Are these two elements equal?
 * @return true 
 * @return false 
 */
bool tools::arrayCompare(const int nObs, bool *h_converge)
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

/**
 * @brief Get a time point (wallclock)
 * 
 * @return double   The time point, in milliseconds
 */
double tools::CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void tools::saveData(const string saveName, int *sets, int nObs)
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

/**
 * @brief Save timing stats
 * 
 * @param timeStats 
 * @param fileName 
 */
void tools::saveStats(double timeStats[6], string fileName)
{
    typedef numeric_limits<double> dbl;

    ofstream timeFile(fileName, ofstream::out | ofstream::app);

    timeFile.precision(dbl::max_digits10);
    for (int i = 0; i < 5; i++)
        timeFile << timeStats[i] << ",";
    timeFile << timeStats[5] << "\n";

    timeFile.close();
}