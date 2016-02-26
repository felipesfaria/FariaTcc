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
	TEST_CLASS(KernelTest)
	{
	public:
		TEST_METHOD(SequentialSvm_i_100Prcnt)
		{
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i"};
			DataSet ds(argc, argv);

			BaseSvm *svm = new SequentialSvm(argc, argv, &ds);

			auto nFolds = 3;
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(&ts, &vs, i);
				svm->Train(&ts);
				svm->Test(&ts,&vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds.nSamples;
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(ParallelSvm_i_100Prcnt)
		{
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i"};
			DataSet ds(argc, argv);

			BaseSvm *svm = new ParallelSvm(argc, argv, &ds);

			auto nFolds = 3;
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(&ts, &vs, i);
				svm->Train(&ts);
				svm->Test(&ts, &vs);
				totalCorrect += vs.nCorrect;
			}
			double expected = 100.0;
			double actual = 100.0*totalCorrect / ds.nSamples;
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(ParallelSvm_a1_GE_70_Prcnt)
		{
			int argc = 3;
			char *argv[] = { "exePath", "-d", "a1" };
			DataSet ds(argc, argv);

			BaseSvm *svm = new ParallelSvm(argc, argv, &ds);

			auto nFolds = 3;
			TrainingSet ts;
			ValidationSet vs;
			int totalCorrect = 0;
			for (auto i = 1; i <= nFolds; i++){
				ds.InitFoldSets(&ts, &vs, i);
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