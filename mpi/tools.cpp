#include <iomanip>
#include <iostream>
#include <random>
#include <time.h>
#include "tools.h"

using namespace std;
using namespace tools;

/**
 * @brief Randomly assign a set to an observation
 * 
 * @param nSets     The number of sets
 * @param sets      The list storing the assign set observations
 * @param x         The observations (might not be necessary)
 */
void tools::randomPartition(int nSets, int *sets, int nObs)
{
    int set0 = 0, set1 = 0, set2 = 0;
    int setN;
    mt19937 rng(1234); // use mersenne twister as rng. seed with const for same results every time
    uniform_int_distribution<mt19937::result_type> dist(0, nSets - 1);

    for (int i = 0; i < nObs; i++)
    {
        setN = dist(rng);
        sets[i] = setN;
        if (setN == 0)
        {
            set0++;
        }
        else if (setN == 1)
        {
            set1++;
        }
        else if (setN == 2)
        {
            set2++;
        }
    }
    cout << "Initial partitioning statistics:" << endl;
    cout << "set0: " << setw(5) << set0 << endl;
    cout << "set1: " << setw(5) << set1 << endl;
    cout << "set2: " << setw(5) << set2 << endl;
    cout << "Total: " << setw(5) << set0 + set1 + set2 << endl;
}

/**
 * @brief Use Forgy method to initialize k-means
 * 
 * @param nSets         The number of sets
 * @param nFeatures     The number of features for each oberservation
 * @param mu            The set means
 * @param nObs          The number of observations in the data
 */
void tools::forgy(int nSets, int nFeatures, float *mu, float x[][5], int nObs)
{
    int val = 0;
    int *obsChoice = new int[nSets]; // the observations to choose from
    for (int i = 0; i < nSets; i++)  // initialize the array
        obsChoice[i] = -1;

    random_device seed;
    mt19937 rng(12345);
    uniform_int_distribution<mt19937::result_type> dist(0, nObs - 1);

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
            int rowIdx = obsChoice[i];  // get the row index from choices
            mu[j + offset] = x[rowIdx][j];
        }
    }

    delete[] obsChoice; // free host memory
}

/**
 * @brief Determine if an element is already in an array
 * 
 * @param arr   The array to check
 * @param item  The item to look for
 * @param n     The number of elements to check (size of array)
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
 * @brief Determine if the two previous set is equivalent to the current set.
 * (As an aside, I'm not sure if there's a better (or more appropriate) way of doing this)
 * 
 * @param nSets         The number of sets
 * @param prevSets      The previous set assignments
 * @param sets          The current set assignments
 */
char tools::arrayCompare(const int nObs, int *prevSets, int *sets)
{
    char equal = 0; // assume the arrays aren't equal, at first

    for (int i = 0; i < nObs; i++)
    {
        equal = prevSets[i] == sets[i]; // see if the elements are equivalent
        if (!equal)                     // if the elements are different, the list isn't equal, so end the loop
            break;
    }

    return equal;
}

void tools::arrayCopy(const int nObs, int *prevSets, int *sets)
{
    for (int i = 0; i < nObs; i++)
    {
        prevSets[i] = sets[i];
    }
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