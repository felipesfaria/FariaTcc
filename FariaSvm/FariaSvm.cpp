#include "stdafx.h"
#include <iterator>
#include <exception>
#include "Logger.h"
#include "DataSet.h"
#include "SequentialSvm.h"
#include "ParallelSvm.cuh"
#include <iostream>
#include "Settings.h"
#include "Utils.h"
#include <ctime>

using namespace std;
int FariaSVM(int argc, char* argv[])
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
			Logger::instance()->Line("Starting Fold " + to_string(i));
			ds.InitFoldSets(ts, vs, i);
			svm->Train(ts);
			Logger::instance()->AddIntMetric("SupportVectors", ts.CountSupportVectors());
			svm->Test(ts, vs);
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
		}
		catch (exception& e2)
		{
			cout << "Base error: " << e.what() << endl;
			cout << "Logger error: " << e2.what() << endl;
		}
		return 1;
	}
}

int main(int argc, char* argv[])
{
	string arg;
	if (!Utils::GetComandVariable(argc, argv, "-auto", arg))
		return FariaSVM(argc, argv);

	vector<char*> args;
	vector<char*> pV = { "1e-2", "1e-4", "1e-8", "1e-12" };
	vector<char*> stV = { "1", "1e-1", "1e-2", "1e-3" };
	vector<char*> smV = { "s", "m" };
	vector<char*> dV = { "a1", "w1" };
	vector<char*> svmV = { "p", "s" };
	vector<char*> tV = { "32","128","512"};
	vector<char*> uaV = { "f", "t" };
	for (auto &p : pV)
		for (auto &st : stV)
			for (auto &sm : smV)
				for (auto &d : dV)
					for (auto &svm : svmV)
						for (auto &t : tV)
							for (auto &ua : uaV)
							{
								if (svm[0] == 'p' && ua[0] == 't') continue;
								if (svm[0] == 's' && t[0] != '3') continue;
								args =
								{
									"FariaSvm.exe",
									"-sd", "0",
									"-d", d,
									"-svm", svm,
									"-f", "3",
									"-mi", "512",
									"-l", "r",
									"-sm", sm,
									"-p", p,
									"-st", st,
									"-ua", ua,
									"-t", t
								};

								for (auto &a : args)
									cout << a << " ";
								cout << endl;
								//FariaSVM(args.size(), args.data());
							};

	return 0;
}