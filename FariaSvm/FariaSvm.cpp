#include "stdafx.h"
#include <iterator>
#include <fstream>
#include <vector>
#include <exception>
#include <ctime>
#include "Logger.h"
#include "DataSet.h"
#include "SequentialSvm.h"
#include "ParallelSvm.cuh"
#include "Utils.h"
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
	try{
		Logger::Init(argc, argv);

		int seed;
		string arg = Utils::GetComandVariable(argc, argv, "-s");
		if (!Utils::TryParseInt(arg, seed))
			seed = time(nullptr);
		Logger::Stats("Seed", seed);
		srand(seed);

		DataSet ds(argc, argv);
		
		arg = Utils::GetComandVariable(argc, argv, "-svm");
		
		BaseSvm *svm;
		if(arg=="p")
			svm = new ParallelSvm(argc, argv, &ds);
		else
			svm = new SequentialSvm(argc, argv, &ds);

		int nFolds;

		arg = Utils::GetComandVariable(argc, argv, "-f");
		if (!Utils::TryParseInt(arg,nFolds))
			nFolds = 3;
		auto totalCorrect = 0;
		int correct;
		for (auto i = 1; i <= nFolds; i++){
			Logger::Fold(i);
			vector<double> alpha1;
			double b1;
			int validationStart = ds.nSamples*(i - 1) / nFolds;
			int validationEnd = ds.nSamples*i / nFolds;
			svm->Train(validationStart, validationEnd, alpha1, b1);
			svm->Test(validationStart, validationEnd, alpha1, b1, correct);
			totalCorrect += correct;
		}
		double averagePercentageCorrect = 100.0*totalCorrect / ds.nSamples;
		Logger::Stats("AveragePercentage", averagePercentageCorrect);
		Logger::End();
		return 0;
	}
	catch (exception& e)
	{
		Logger::Error(e);
		return 1;
	}
}