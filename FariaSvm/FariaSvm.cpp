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
		Settings::instance()->Init(argc, argv);
		auto m = Logger::instance()->StartMetric("main");
		unsigned seed;
		Settings::instance()->GetUnsigned("seed", seed);
		srand(seed);

		DataSet ds;
		
		BaseSvm *svm = BaseSvm::GenerateSvm(ds);

		Logger::instance()->LogSettings();

		TrainingSet ts;
		ValidationSet vs;
		for (auto i = 1; i <= ds.nFolds; i++){
			Logger::instance()->Line("Starting Fold "+to_string(i));
			ds.InitFoldSets(ts, vs, i);
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
		try{
			Logger::instance()->Error(e);
		} catch (exception& e2)
		{
			cout << "Base error: " << e.what() << endl;
			cout << "Logger error: " << e2.what() << endl;
		}
		return 1;
	}
}