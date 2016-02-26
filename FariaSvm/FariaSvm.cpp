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
		string arg = Utils::GetComandVariable(argc, argv, "-sd");
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

		TrainingSet ts;
		ValidationSet vs;
		auto totalCorrect = 0;
		for (auto i = 1; i <= ds.nFolds; i++){
			Logger::Fold(i);
			ds.InitFoldSets(&ts, &vs, i);
			svm->Train(&ts);
			svm->Test(&ts,&vs);
			totalCorrect += vs.nCorrect;
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