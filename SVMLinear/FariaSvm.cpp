// SVMLinear.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <exception>
#include <ctime>
#include "Logger.h"
#include "Utils.h"
#include <algorithm>
#include "DataSet.h"
#include "SvmLinear.h"

using namespace std;

void ReadFile(DataSet& ds)
{
	Logger::FunctionStart("ReadFile");
	ifstream       file;
	file.open(ds.FileName, ifstream::in);

	if (!file.good())
		throw(new exception("Error: File not found"));
	ds.ReadFile(file);
	Logger::FunctionEnd();
}

int main(int argc, char* argv[])
{
	try{
		unsigned int seed = time(nullptr);
		Logger::Init(argc, argv);
		Logger::Stats("Seed",seed);
		srand(seed);

		DataSet ds(argc, argv);

		ReadFile(ds);
		SvmLinear svm;
		Utils::Shuffle(ds.X, ds.Y);

		double validationPercentage = 0.5;
		int nValidators = (ds.X.size()*validationPercentage);
		int nTrainers = ds.X.size() - nValidators;
		LinearKernel kernel;
		kernel.DefineGaussian(ds.Gama);
		//kernel.DefineHomogeneousPolynomial(2);
		int iFold = 1;
		
		Logger::Fold(iFold++);

		vector<double> alpha1;
		double b1;
		int nCorrect1;
		svm.Train(ds, nTrainers, kernel, alpha1, b1);
		svm.Test(ds, nValidators, nTrainers, kernel, alpha1, b1, nCorrect1);

		Utils::Reverse(ds.X, ds.Y);

		Logger::Fold(iFold++);

		vector<double> alpha2;
		double b2;
		int nCorrect2;
		svm.Train(ds, nTrainers, kernel, alpha2, b2);
		svm.Test(ds, nValidators, nTrainers, kernel, alpha2, b2, nCorrect2);

		double totalCorrect = nCorrect1 + nCorrect2;
		double totalSamples = nValidators + nTrainers;
		double averagePercentageCorrect = totalCorrect / totalSamples;
		Logger::Percentage(totalCorrect, totalSamples, averagePercentageCorrect,"Average ");
		Logger::End();
		return 0;
	}
	catch (exception& e)
	{
		Logger::Error(e);
		return 1;
	}
}