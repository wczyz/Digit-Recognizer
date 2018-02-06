#ifndef MISC_H
#define MISC_H

/**
*	File storing declarations of miscellaneous functions.
**/

#include <math.h>
#include <cstdlib>
#include <ctime>

// Sigmoid function.
double sigmoid(double);

// Derivative of the sigmoid function.
double sigmoidPrime(double);

// Function returning a random double value in range [fMin, fMax].
double fRand(double fMin, double fMax);	

#endif // !MISC_H
