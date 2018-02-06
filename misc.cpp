/**
*	File storing implementations of miscellaneous functions.
**/

#include "misc.h"

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double sigmoidPrime(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));	
}

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}