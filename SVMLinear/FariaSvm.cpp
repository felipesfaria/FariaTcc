#include "stdafx.h"
#include <iterator>
#include <fstream>
#include <vector>
#include <exception>
#include <ctime>
#include "Logger.h"
#include "Utils.h"
#include <algorithm>
#include "DataSet.h"
#include "SvmLinear.h"
#include "SequentialKernel.h"

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
		Logger::Stats("Seed", seed);
		srand(seed);

		DataSet ds(argc, argv);

		ReadFile(ds);
		SvmLinear svm;
		Utils::Shuffle(ds.X, ds.Y);

		svm.kernel = new SequentialKernel();
		svm.kernel->Init(ds);
		auto nFolds = 3;
		auto totalCorrect = 0;
		int correct;
		for (auto i = 1; i <= nFolds; i++){
			Logger::Fold(i);
			vector<double> alpha1;
			double b1;
			int validationStart = ds.nSamples*(i - 1) / nFolds;
			int validationEnd = ds.nSamples*i / nFolds;
			svm.Train(ds, validationStart, validationEnd, alpha1, b1);
			svm.Test(ds, validationStart, validationEnd, alpha1, b1, correct);
			totalCorrect += correct;
		}
		Utils::Reverse(ds.X, ds.Y);
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