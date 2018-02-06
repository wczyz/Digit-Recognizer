/**
*	This project is about creating a neural network to recognize handwritten digits.
*	As a dataset I'm using MNIST dataset. Link: http://yann.lecun.com/exdb/mnist/.
*	I'm accesing it thanks to "Simple C++ reader for MNIST dataset", which can be found here: https://github.com/wichtounet/mnist.
**/

#include <iostream>
#include <vector>
#include <iomanip>
#include "mnist/mnist_reader.hpp"
#include "settings.hpp"
#include "network.h"

int main()
{
	// Initialization of a pseudo-random number generator.
	srand(time(NULL));

	// Loading MNIST data.
	std::cout << std::fixed << std::setprecision(3);
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

	// Vectors storing values from both training and test datasets as doubles from range [0.0, 1.0].
	std::vector < std::vector <double> > trainingImages;
	std::vector < std::vector <double> > testImages;

	// Vectors storing expected values for each entry from the dataset.
	std::vector < std::vector <double> > trainingExpected;
	std::vector <int> testExpected;

	// Auxiliary 1d doubles vector.
	std::vector <double> vec1;

	// Checking whether the dataset was loaded properly.
	if (!dataset.training_images.empty())
		std::cout << "Loading successful." << std::endl;
	else
		std::cout << "Error: MNIST dataset couldn't load." << std::endl;

	// Normalazing the dataset. Original dataset is a vector of grayscale images - integer vectors with values from 0 to 255.
	// I'm converting it to the range [0.0, 1.0].
	for (int i = 0; i < DATASET_SIZE; i++)
	{
		trainingImages.push_back(vec1);
		for (int j = 0; j < 28 * 28; j++)
			trainingImages[i].push_back((double)dataset.training_images[i][j] / 255);
	}

	for (int i = 0; i < dataset.test_images.size(); i++)
	{
		testImages.push_back(vec1);
		for (int j = 0; j < 28 * 28; j++)
			testImages[i].push_back((double)dataset.test_images[i][j] / 255);
	}


	// Setting the "expected" vectors.
	for (int i = 0; i < dataset.test_labels.size(); i++)
		testExpected.push_back((int)dataset.test_labels[i]);

	for (int i = 0; i < DATASET_SIZE; i++)
	{
		trainingExpected.push_back(vec1);
		for (int j = 0; j < 10; j++)
			if (dataset.training_labels[i] == j)
				trainingExpected[i].push_back(1.0);
			else
				trainingExpected[i].push_back(0.0);
	}

	// Neural net object initialization, setting the dataset and performing gradient descend and evaluation.
	int layers[] = { 28 * 28, 30, 10 };
	Network net(3, layers);
	net.setDataset(trainingImages, trainingExpected, DATASET_SIZE);

	net.gradientDescent(0.75, DATASET_SIZE);	
	net.evaluate(20, testImages, testExpected);
}