#ifndef NETWORK_H
#define NETWORK_H

/**
*	File with network class' declaration.
**/

#include <iostream>
#include <vector>
#include <iomanip>
#include "misc.h"

class Network
{
private:

	// Number of layers and vector storing number of neurons in each layer.
	int networkSize;
	std::vector <int> layer;

	// Vectors storing neural net's weights, biases, activation values, and activation values without activation function applied.
	std::vector < std::vector < std::vector <double> > > weight;
	std::vector < std::vector <double> > bias;
	std::vector < std::vector <double> > activation;
	std::vector < std::vector <double> > value;

	// Dataset size.
	int datasetSize;
	// Vectors storing the dataset and expected values.
	std::vector < std::vector <double> > dataset;
	std::vector < std::vector <double> > expected;

public:

	// Constructor.
	Network(int n, int *nums);

	// Function setting	neural net's dataset and expected vectors and their sizes.
	void setDataset(std::vector < std::vector <double> > &dataset, std::vector < std::vector <double> > &expected, int size);

	// Function that performs a feed-forward procedure.
	void feedForward();

	// Function that returns activation value of i-th neuron on the output layer.
	double getScore(int i);

	// Function that sets input layer's activation values.
	void input(std::vector <double> x);

	// Function performing online gradient descent algorithm with learning rate equal to alpha and number of iterations equal to N,
	void gradientDescent(double alpha, int N);

	// Function evaluating how good is this neural network at reading handwritten digits.
	void evaluate(int n, std::vector < std::vector <double> > &data, std::vector <int> &exp);
};

#endif // !NETWORK_H
