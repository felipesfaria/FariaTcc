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

		DataSet ds;

		string svmType;
		Settings::instance()->GetString("svm", svmType);
		
		BaseSvm *svm;
		if (svmType == "p")
			svm = new ParallelSvm(&ds);
		else
			svm = new SequentialSvm(&ds);

		Logger::instance()->LogSettings();

		TrainingSet ts;
		ValidationSet vs;
		for (auto i = 1; i <= ds.nFolds; i++){
			Logger::instance()->Fold(i);
			ds.InitFoldSets(&ts, &vs, i);
			svm->Train(&ts);
			Logger::instance()->AddIntMetric("SupportVectors", ts.CountSupportVectors());
			svm->Test(&ts, &vs);
			Logger::instance()->AddDoubleMetric("PercentCorrect", vs.GetPercentage());
			Logger::instance()->AddIntMetric("Correct", vs.nCorrect);
			Logger::instance()->AddIntMetric("NullWrong", vs.nNullWrong);
			Logger::instance()->AddIntMetric("PositiveWrong", vs.nPositiveWrong);
			Logger::instance()->AddIntMetric("NegativeWrong", vs.nNegativeWrong);
		}
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