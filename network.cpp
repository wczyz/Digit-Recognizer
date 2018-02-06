/**
*	File with network class' implementation.
**/

#include "network.h"

Network::Network(int n, int *nums)
{
	// Setting the number of layers and number of neurons in each layer.
	networkSize = n;
	for (int i = 0; i < n; i++)
		layer.push_back(nums[i]);

	// Double variable, 1d and 2d double vectors for further use.
	double x;
	std::vector <double> vec1;
	std::vector < std::vector <double> > vec2;

	// Setting random values for: weight, bias and zero values for activation and value.

	// Weight
	for (int i = 0; i < networkSize - 1; i++)
	{
		weight.push_back(vec2);
		for (int j = 0; j < layer[i]; j++)
		{
			weight[i].push_back(vec1);
			for (int k = 0; k < layer[i + 1]; k++)
				weight[i][j].push_back(fRand(-1.0, 1.0));
		}
	}

	// Bias
	for (int i = 0; i < networkSize; i++)
	{
		bias.push_back(vec1);
		for (int j = 0; j < layer[i]; j++)
			bias[i].push_back(fRand(-1.0, 1.0));
	}

	// Activation
	for (int i = 0; i < networkSize; i++)
	{
		activation.push_back(vec1);
		for (int j = 0; j < layer[i]; j++)
			activation[i].push_back(0.0);
	}

	// Value
	for (int i = 0; i < networkSize; i++)
	{
		value.push_back(vec1);
		for (int j = 0; j < layer[i]; j++)
			value[i].push_back(0.0);
	}

}

void Network::setDataset(std::vector < std::vector <double> > &data, std::vector < std::vector <double> > &exp, int size)
{
	datasetSize = size;

	dataset = data;
	expected = exp;
}

void Network::feedForward()
{
	for (int i = 1; i < networkSize; i++)
	{
		for (int j = 0; j < layer[i]; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < layer[i - 1]; k++)
				sum += activation[i - 1][k] * weight[i - 1][k][j];

			sum += bias[i][j];

			value[i][j] = sum;
			activation[i][j] = sigmoid(value[i][j]);
		}
	}
}

double Network::getScore(int i)
{
	return activation[networkSize - 1][i];
}

void Network::input(std::vector <double> x)
{
	for (int i = 0; i < x.size(); i++)
		activation[0][i] = x[i];
}

void Network::gradientDescent(double alpha, int N)
{
	std::cout << "Start of the gradient descent!" << std::endl;

	// Vectors storing derivatives of the cost function with respect to biases and with respect to weights.
	std::vector <double> derivativeB1, derivativeB2;
	std::vector < std::vector <double> > derivativeW0, derivativeW1;

	// 1d doubles vector for further use.
	std::vector <double> vec1;

	// Filling vectors with zeros.
	for (int i = 0; i < layer[1]; i++)
		derivativeB1.push_back(0.0);

	for (int i = 0; i < layer[2]; i++)
		derivativeB2.push_back(0.0);

	for (int i = 0; i < layer[0]; i++)
	{
		derivativeW0.push_back(vec1);
		for (int j = 0; j < layer[1]; j++)
			derivativeW0[i].push_back(0.0);
	}

	for (int i = 0; i < layer[1]; i++)
	{
		derivativeW1.push_back(vec1);
		for (int j = 0; j < layer[2]; j++)
			derivativeW1[i].push_back(0.0);
	}

	for (int z = 0; z < N; z ++)
	{
		double sum = 0.0;
		double x = 0.0;

		int i = z % datasetSize;

		input(dataset[i]);
		feedForward();

		x = 0.0;

		// Bias_2 derivative.
		for (int a = 0; a < layer[2]; a++)
		{
			x = (activation[2][a] - expected[i][a]) * sigmoidPrime(value[2][a]);
			derivativeB2[a] = x;
		}

		// Weight_1 derivative.
		for (int a = 0; a < layer[1]; a++)
		{
			for (int b = 0; b < layer[2]; b++)
			{
				x = (activation[2][b] - expected[i][b]) * sigmoidPrime(value[2][b]) * activation[1][a];
				derivativeW1[a][b] = x;
			}
		}

		// Bias_1 derivative.
		for (int a = 0; a < layer[1]; a++)
		{
			sum = 0.0;
			for (int j = 0; j < layer[2]; j++)
			{
				x = 0.0;
				x = (activation[2][j] - expected[i][j]) * sigmoidPrime(value[2][j]) * weight[1][a][j] * sigmoidPrime(value[1][a]);
				sum += x;
			}
			derivativeB1[a] = sum;
		}

		// Weight_0 derivative.
		for (int a = 0; a < layer[0]; a++)
		{
			for (int b = 0; b < layer[1]; b++)
			{
				sum = 0.0;
				for (int j = 0; j < layer[2]; j++)
				{
					x = 0.0;
					x = (activation[2][j] - expected[i][j]) * sigmoidPrime(value[2][j]) * weight[1][b][j] * sigmoidPrime(value[1][b]) * activation[0][a];
					sum += x;
				}
				derivativeW0[a][b] = sum;
			}
		}

		// Here actual gradient descent is applied. Values of weights and biases are changed by a fraction of its derivatives.
		for (int i = 0; i < layer[2]; i++)
			bias[2][i] -= alpha * derivativeB2[i];

		for (int i = 0; i < layer[1]; i++)
			bias[1][i] -= alpha * derivativeB1[i];

		for (int i = 0; i < layer[1]; i++)
			for (int j = 0; j < layer[2]; j++)
				weight[1][i][j] -= alpha * derivativeW1[i][j];

		for (int i = 0; i < layer[0]; i++)
			for (int j = 0; j < layer[1]; j++)
				weight[0][i][j] -= alpha * derivativeW0[i][j];

		if ((z % 100) == 0)
			std::cout << std::endl << std::endl << "Iteration nr " << z + 1 << std::endl << std::endl;
	}
}

void Network::evaluate(int n, std::vector < std::vector <double> > &data, std::vector <int> &exp)
{
	std::cout << "Start of evaluation." << std::endl;

	// Showing net's performance on the 20 first cases from the test dataset.
	for (int i = 0; i < n; i++)
	{
		std::cout << "Dataset nr: " << i + 1 << std::endl;
		
		// Loading dataset entry to the neural net.
		input(data[i]);
		feedForward();

		// Converting expected vector element to a double vector.
		std::vector <double> x;
		for (int j = 0; j < layer[2]; j++)
		{
			if (exp[i] != j)
				x.push_back(0.0);
			else
				x.push_back(1.0);
		}

		for (int j = 0; j < layer[2]; j++)
			std::cout << j << ": " << activation[2][j] << "     " << x[j] << std::endl;

		
	}

	// Calculating net's overall performence on the test dataset.

	int sum = 0;
	double maxA, maxB;
	int maxA_i, maxB_i;

	for (int i = 0; i < data.size(); i++)
	{
		if ((i % 100) == 0)
			std::cout << "Evaluation nr " << i + 1 << std::endl;

		maxA = 0.0;
		maxA_i = -1;
		input(data[i]);
		feedForward();

		for (int j = 0; j < 10; j++)
		{
			if (activation[2][j] > maxA)
			{
				maxA = activation[2][j];
				maxA_i = j;
			}
		}

		if (maxA_i == exp[i])
			sum++;
	}
	double score = (double)sum / data.size();

	std::cout << "Scored " << sum << " out of " << data.size() << std::endl;
	std::cout << score * 100 << "%" << std::endl;
	
}