#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/SequentialSvm.h"
#include "../FariaSvm/ParallelSvm.cuh"
#include "../FariaSvm/Utils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
namespace FariaTccTest
{
	TEST_CLASS(SvmTest)
	{
	public:
		TEST_METHOD(SequentialSvm_i_100Prcnt)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-svm", "s"
				"-f", "3",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;

			BaseSvm *svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(ts, vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds.nSamples;
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
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;

			BaseSvm *svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(ts, vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds.nSamples;
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
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;

			BaseSvm *svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(ts, vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds.nSamples;
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
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;

			BaseSvm *svm = BaseSvm::GenerateSvm(ds);

			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(ts, vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds.nSamples;
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(Compare_Parallel_and_Sequential_Alpha)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-st", "0.1",
				"-mi", "1024",
				"-sd", "0",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;

			BaseSvm *sSvm = BaseSvm::GenerateSvm(ds, "s");
			TrainingSet sTs;
			ValidationSet sVs;
			ds.InitFoldSets(sTs, sVs, 1);
			sSvm->Train(&sTs);

			BaseSvm *pSvm = BaseSvm::GenerateSvm(ds, "p");
			TrainingSet pTs;
			ValidationSet pVs;
			ds.InitFoldSets(pTs, pVs, 1);
			pSvm->Train(&pTs);

			int expected = 0;
			int actual = 0;
			for (int i = 0; i < pTs.height; ++i)
				if (abs(pTs.alpha[i] - sTs.alpha[i]) > 1e-10)
					actual++;
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(Compare_Parallel_and_Sequential_Alpha_MultiStep)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-st", "0.1",
				"-mi", "1024",
				"-sd", "0",
				"-l", "n",
				"-sm", "m"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;

			BaseSvm *sSvm = BaseSvm::GenerateSvm(ds, "s");
			TrainingSet sTs;
			ValidationSet sVs;
			ds.InitFoldSets(sTs, sVs, 1);
			sSvm->Train(&sTs);

			BaseSvm *pSvm = BaseSvm::GenerateSvm(ds, "p");
			TrainingSet pTs;
			ValidationSet pVs;
			ds.InitFoldSets(pTs, pVs, 1);
			pSvm->Train(&pTs);

			int expected = 0;
			int actual = 0;
			for (int i = 0; i < pTs.height; ++i)
				if (abs(pTs.alpha[i] - sTs.alpha[i]) > 1e-10)
					actual++;
			Assert::AreEqual(expected, actual);
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
			DataSet ds;

			BaseSvm *svm = BaseSvm::GenerateSvm(ds);

			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(ts, vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = true;
			double actual = 100.0*totalCorrect / ds.nSamples > 70.0;
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
			DataSet ds;

			BaseSvm *svm = BaseSvm::GenerateSvm(ds);

			unsigned nFolds;
			Settings::instance()->GetUnsigned("folds", nFolds);
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(ts, vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = true;
			double actual = 100.0*totalCorrect / ds.nSamples > 70.0;
			Assert::AreEqual(expected, actual);
		}

	};
}