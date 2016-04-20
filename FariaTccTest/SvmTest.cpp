#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/SequentialSvm.h"
#include "../FariaSvm/ParallelSvm.cuh"
#include "../FariaSvm/Utils.h"
#include <memory>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace FariaSvm;
namespace FariaTccTest
{
	TEST_CLASS(SvmTest)
	{
	public:
		TEST_CLASS_CLEANUP(CleanUpSvmTest)
		{
			FariaSvm::Logger::Delete();
			Settings::Delete();
		}
		TEST_METHOD(SequentialSvm_i_100Prcnt)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-svm", "s"
				"-f", "3",
				"-sd", "0",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds->InitFoldSets(ts, vs, i);
				svm->Train(ts);
				svm->Test(ts, vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds->nSamples;
			Assert::AreEqual(expected, actual);


		}

		TEST_METHOD(SequentialSvm_i_100PrcntMultiStep)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-svm", "s",
				"-f", "3",
				"-l", "n",
				"-sd", "0",
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds->InitFoldSets(ts, vs, i);
				svm->Train(ts);
				svm->Test(ts, vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds->nSamples;
			Assert::AreEqual(expected, actual);


		}

		TEST_METHOD(ParallelSvm_i_100Prcnt)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-svm", "p"
				"-f", "3",
				"-sd", "0",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds->InitFoldSets(ts, vs, i);
				svm->Train(ts);
				svm->Test(ts, vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds->nSamples;
			Assert::AreEqual(expected, actual);


		}

		TEST_METHOD(ParallelSvm_i_100Prcnt_MultiStep)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-svm", "p"
				"-f", "3",
				"-l", "n",
				"-sd", "0",
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds->InitFoldSets(ts, vs, i);
				svm->Train(ts);
				svm->Test(ts, vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds->nSamples;
			Assert::AreEqual(expected, actual);


		}

		TEST_METHOD(Compare_Parallel_and_Sequential_Alpha)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-st", "1",
				"-mi", "1024",
				"-sd", "0",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto sSvm = BaseSvm::GenerateSvm(ds, "s");
			TrainingSet sTs;
			ValidationSet sVs;
			ds->InitFoldSets(sTs, sVs, 1);
			sSvm->Train(sTs);

			auto pSvm = BaseSvm::GenerateSvm(ds, "p");
			TrainingSet pTs;
			ValidationSet pVs;
			ds->InitFoldSets(pTs, pVs, 1);
			pSvm->Train(pTs);

			int nErrors = 0;
			double avgError = 0;
			for (int i = 0; i < pTs.height; ++i)
			{
				auto dif = abs(pTs.alpha[i] - sTs.alpha[i]);
				if (dif){
					nErrors++;
					avgError += dif;
				}
			}
			if (nErrors)
				avgError /= nErrors;

			double avgErrorTolerance = 1e-14;
			Assert::IsTrue(avgError < avgErrorTolerance);



		}

		TEST_METHOD(Compare_Parallel_and_Sequential_Classification)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "a1",
				"-st", "0.01",
				"-p", "0.01",
				"-mi", "3",
				"-sd", "0",
				"-l", "n",
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto sequentialSvm = BaseSvm::GenerateSvm(ds, "s");
			TrainingSet sequentialTrainingSet;
			ValidationSet sequentialValidationSet;
			ds->InitFoldSets(sequentialTrainingSet, sequentialValidationSet, 1);
			sequentialSvm->Train(sequentialTrainingSet);

			auto parallelSvm = BaseSvm::GenerateSvm(ds, "p");
			TrainingSet parallelTrainingSet;
			ValidationSet parallelValidationSet;
			ds->InitFoldSets(parallelTrainingSet, parallelValidationSet, 1);
			parallelSvm->Train(parallelTrainingSet);

			int actual = 0;
			for (int i = 0; i < parallelValidationSet.height; ++i)
			{
				auto parallelClassified = parallelSvm->Classify(parallelTrainingSet, parallelValidationSet, i);
				auto sequentialClassified = sequentialSvm->Classify(sequentialTrainingSet, sequentialValidationSet, i);
				if (parallelClassified != sequentialClassified)
					actual++;
			}
			int expected = 0;
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(Compare_Parallel_and_Sequential_Alpha_MultiStep)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-st", "1",
				"-mi", "1024",
				"-sd", "0",
				"-l", "n"
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto sSvm = BaseSvm::GenerateSvm(ds, "s");
			TrainingSet sTs;
			ValidationSet sVs;
			ds->InitFoldSets(sTs, sVs, 1);
			sSvm->Train(sTs);

			auto pSvm = BaseSvm::GenerateSvm(ds, "p");
			TrainingSet pTs;
			ValidationSet pVs;
			ds->InitFoldSets(pTs, pVs, 1);
			pSvm->Train(pTs);

			int nErrors = 0;
			double avgError = 0;
			for (int i = 0; i < pTs.height; ++i)
			{
				auto dif = abs(pTs.alpha[i] - sTs.alpha[i]);
				if (dif){
					nErrors++;
					avgError += dif;
				}
			}
			if (nErrors)
				avgError /= nErrors;

			double avgErrorTolerance = 1e-14;
			Assert::IsTrue(avgError < avgErrorTolerance);
		}

		TEST_METHOD(ParallelSvm_a1_GE_70_Prcnt)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "a1",
				"-svm", "p",
				"-f", "2",
				"-mi", "4",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto svm = BaseSvm::GenerateSvm(ds);

			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds->InitFoldSets(ts, vs, i);
				svm->Train(ts);
				svm->Test(ts, vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = true;
			double actual = 100.0*totalCorrect / ds->nSamples > 70.0;
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(ParallelSvm_a1_GE_70_Prcnt_MultiStep)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "a1",
				"-svm", "p",
				"-f", "2",
				"-mi", "4",
				"-l", "n",
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			auto ds = make_shared<DataSet>();

			auto svm = BaseSvm::GenerateSvm(ds);

			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds->InitFoldSets(ts, vs, i);
				svm->Train(ts);
				svm->Test(ts, vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = true;
			double actual = 100.0*totalCorrect / ds->nSamples > 70.0;
			Assert::AreEqual(expected, actual);


		}

	};
}