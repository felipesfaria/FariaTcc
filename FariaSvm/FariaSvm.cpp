#include "stdafx.h"
#include <iterator>
#include <exception>
#include "Logger.h"
#include "DataSet.h"
#include "SequentialSvm.h"
#include "ParallelSvm.cuh"
#include <iostream>
#include "Settings.h"

using namespace std;
int main(int argc, char* argv[])
{
	try{
		auto m = Logger::instance()->StartMetric("main");
		Settings::instance()->Init(argc, argv);
		unsigned seed;
		Settings::instance()->GetUnsigned("seed", seed);
		srand(seed);

		DataSet ds(argc, argv);

		string svmType;
		Settings::instance()->GetString("svm", svmType);
		
		BaseSvm *svm;
		if (svmType == "p")
			svm = new ParallelSvm(argc, argv, &ds);
		else
			svm = new SequentialSvm(argc, argv, &ds);

		Logger::instance()->LogSettings();

		TrainingSet ts;
		ValidationSet vs;
		auto totalCorrect = 0;
		for (auto i = 1; i <= ds.nFolds; i++){
			Logger::instance()->Fold(i);
			ds.InitFoldSets(&ts, &vs, i);
			svm->Train(&ts);
			svm->Test(&ts,&vs);
			totalCorrect += vs.nCorrect;
		}
		double averagePercentageCorrect = 100.0*totalCorrect / ds.nSamples;
		Logger::instance()->Stats("AveragePercentage", averagePercentageCorrect);
		m->Stop();
		Logger::instance()->End();
		delete(svm);
		return 0;
	}
	catch (exception& e)
	{
		Logger::instance()->Error(e);
		return 1;
	}
}