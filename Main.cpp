#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

//Number of cycles to train on
const int numberOfEpochs = 1000;
//Number of Nodes in each layer
//3 layered approach, can have multiple intermittent hidden layers if required
const int numInput = 1;
const int numHidden = 2;
const int numOutput = 1;
//Number of Training Patterns
const int pattern = 1000;
//Number of data values to train against
const int training = 300;
//Number of data values to test against
const int testing = 92;
const double bias = 0.9;


//Function to determine associated output with any given input
double sigmoid(double in)
{
	return 1.0 / (1.0 + exp(-in));
}

double sigDerivative(double sigmoid)
{
	return sigmoid * (1.0 - sigmoid);
}

double random()
{
	return ((double)rand() / ((double)RAND_MAX + 1));
}



void fileInput(double(&input)[pattern + 1][(numInput + 1)], double(&target)[pattern + 1][(numOutput + 1)])
{
	//Gather Data from File
	//Note: This form of input only works with timeseries data in the form of 2 columns.
	ifstream dataFile("data.txt");
	//Data Accessible?
	if (dataFile) {
		//For each line in file, the length of our overall dataset (training + testing)
		for (int x = 0; x < testing + training; x++)
		{
			double iValue;
			double tValue;
			dataFile >> iValue >> tValue;

			input[x][0] = 0;
			input[x][1] = iValue;
			target[x][0] = 0;
			target[x][1] = tValue;
		}
	}
	else
	{
		cout << "File not found";
	}
}


int main()
{
	double targetError = 0.1;
	double error = 999.0;
	int randomPattern[training+1];
	double input[pattern + 1][numInput + 1];
	double target[pattern + 1][numOutput + 1];
	double hidden[pattern+ 1][numHidden + 1];
	double output[pattern + 1][numOutput + 1];
	
	double sumH[pattern + 1][numHidden + 1];//Sum of Input * WeightIH
	double sumO[pattern + 1][numOutput + 1]; //Sum of Hidden * WeightHO + WeightHO
	double SumDOW[numHidden + 1]; //Sum of Delta Output Weights

	double weightIH[numInput + 1][numHidden + 1];
	double weightHO[numHidden + 1][numOutput + 1];
	
	
	double deltaH[numHidden+1]; // Difference between target and hidden value
	double deltaO[numOutput+1]; // Difference between target and output value
	double deltaWeightIH[numInput + 1][numHidden + 1];
	double deltaWeightHO[numHidden + 1][numOutput + 1];

	double eta = 0.5; // Learning Rate
	double alpha = 0.9; //momentum
	fileInput(input, target);

	//Initialize Weights (IH, HO, DeltaIH, DeltaHO)
	for (int h = 1; h <= numHidden; h++) {
		for (int i = 0; i <= numInput; i++) {
			deltaWeightIH[i][h] = 0.0; //Could initialize it with the array, doing it now allows for extendability
				; //Bias labelled smallwt by john.
		}
	}
	for (int o = 1; o <= numOutput; o++) {
		for (int h = 0; h <= numHidden; h++) {
			deltaWeightHO[h][o] = 0.0; //Could initialize it with the array, doing it now allows for extendability
			weightHO[h][o] = 2.0 * (random() - 0.5) * bias;
		}
	}

	for (int epoch = 0; epoch < numberOfEpochs; epoch++) //Training loop for the NN, using only the first 300 data items.
	{
		error = 0;

		//for (int p = 0; p <= training; p++) { //
		//	randomPattern[p] = p; //Secondary pattern array, for mixing up the order in which values will appear
		//}
		//for (int p = 0; p <= training; p++) { //Shuffle order of patterns For-Loop. Useful for avoiding local minima.
		//	int swapValue = p + random() * (training + 1 - p);
		//	int originalPattern = randomPattern[p];
		//	randomPattern[p] = randomPattern[swapValue];
		//	randomPattern[swapValue] = originalPattern;
		//}
		
		for (int i = 0; i < training; i++) //For Each of our new patterns:
		{ 
			int a = i;// randomPattern[i]; // Shortened form for loops below

			for (int j = 1; j <= numHidden; j++) //Compute hidden activations (movement between input and hidden layers)
			{
				sumH[a][j] = weightIH[0][j]; //Input -> Hidden Bias initialized earlier (after fileInput() )
				for (int k = 1; k <= numInput; k++)  //For number of input nodes 
				{
					sumH[a][j] += input[a][k] * weightIH[k][j]; // Sum increased by input weights and bias weight
				}
				hidden[a][j] = sigmoid(sumH[a][j]); // Sigmoid Function to update hidden node.
			}

			for (int j = 1; j <= numOutput; j++)//Compute Output activations (Movement between hidden and output layers)
			{
				sumO[a][j] = weightHO[0][j];//Hidden -> Output Bias initialized earlier (after fileInput() )
				for (int k = 1; k <= numHidden; k++)
				{
					sumO[a][j] += hidden[a][k] * weightHO[k][j];
				}
				output[a][j] = sigmoid(sumO[a][j]); // Sigmoid Function to update output node.
				error += 0.5 * (target[a][j] - output[a][j]) * (target[a][j] - output[a][j]); //Sum of Squares Error
				deltaO[j] = (target[a][j] - output[a][j]) * output[a][j] * (1.0 - output[a][j]); // Calculate difference between output and target
				//Should this line be * sigmoid(output[a][j]??
			}

			//Back-Propogation of errors
			for (int j = 1; j <= numHidden; j++)
			{
				SumDOW[j] = 0.0; //Reset each epoch
				for (int k = 1; k <= numOutput; k++)
				{
					SumDOW[j] += weightHO[j][k] * deltaO[k]; //Weight of each Hidden -> Output connection multiplied by the difference between Output and target
				}
				deltaH[j] = SumDOW[j] * hidden[a][j] * (1.0 - hidden[a][j]); // 
			}
			//Update weights IH
			for (int j = 1; j <= numHidden; j++) 
			{
				deltaWeightIH[0][j] = eta * deltaH[j] + alpha * deltaWeightIH[0][j]; //Delta Weight shifts based off of deltaH, learning rate and momentum.
				weightIH[0][j] += deltaWeightIH[0][j]; //weights Input -> Hidden adjusted based off the shift from the delta value
				for (int k = 1; k <= numInput; k++)
				{
					deltaWeightIH[k][j] = eta * input[a][k] * deltaH[j] + alpha * deltaWeightIH[k][j];
					weightIH[k][j] += deltaWeightIH[k][j];
				}
			}
			//Update weights HO
			for (int j = 1; j <= numOutput; j++)
			{
				deltaWeightHO[0][j] = eta * deltaO[j] + alpha * deltaWeightHO[0][j]; //weights Hidden -> Ouput adjusted based off the shift from the delta value
				weightHO[0][j] += deltaWeightHO[0][j];
				for (int k = 1; k <= numHidden; k++)
				{
					deltaWeightHO[k][j] = eta * hidden[a][k] * deltaO[j] + alpha * deltaWeightHO[k][j];
					weightHO[k][j] += deltaWeightHO[k][j];
				}
			}
		}
		cout << "Epoch: " << epoch+1 << " | Error: " << error << "\n";
		if (error < targetError)
		{
			break;
		}
	}


	ofstream outputFile("results.txt");
	if (!outputFile)
	{
		cout << "Error, no output file found!";
		return 1;
	}


	for (int p = training; p <= training + testing; p++) //Test the System on all the remaining  testing data (last 92)
	{
			for (int j = 1; j <= numHidden; j++) //Compute hidden activations (movement between input and hidden layers)
			{
				sumH[p][j] = weightIH[0][j]; //Input -> Hidden Bias initialized earlier (after fileInput() )
				for (int k = 1; k <= numInput; k++)  //For number of input nodes 
				{
					sumH[p][j] += input[p][k] * weightIH[k][j]; // Sum increased by input weights and bias weight
				}
				hidden[p][j] = sigmoid(sumH[p][j]); // Sigmoid Function to update hidden node.
			}

			for (int j = 1; j <= numOutput; j++)//Compute Output activations (Movement between hidden and output layers)
			{
				sumO[p][j] = weightHO[0][j];//Hidden -> Output Bias initialized earlier (after fileInput() )
				for (int k = 1; k <= numHidden; k++)
				{
					sumO[p][j] += hidden[p][k] * weightHO[k][j];
				}
				output[p][j] = sigmoid(sumO[p][j]); // Sigmoid Function to update output node.
				error += 0.5 * (target[p][j] - output[p][j]) * (target[p][j] - output[p][j]); //Sum of Squares Error
			}
			//Do File Output here
			//MSE = Error / Total (training + testing)
			error = error / (training + testing);
			outputFile << "Mean Squared Error : " << error << endl;
	}
	for (int x = 1; x <= training + testing; x++) {
		//Output the difference between each Ouput node and its relevant target for comparison
		for (int y = 1; y <= numOutput; y++)
		{
			outputFile << "Output: " << output[x][y] << "\t Target: " << target[x][y] << endl;
		}

	}


	
	system("pause");
	return 0;
}