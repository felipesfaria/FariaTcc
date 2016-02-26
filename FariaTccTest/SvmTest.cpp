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
			char *argv[] = { "exePath", "-d", "i","-s","p" };
			DataSet ds(argc, argv);

			BaseSvm *svm = new SequentialSvm(argc, argv, &ds);

			auto nFolds = 3;
			auto totalCorrect = 0;
			int correct;
			for (auto i = 1; i <= nFolds; i++){
				vector<double> alpha1;
				double b1;
				int validationStart = ds.nSamples*(i - 1) / nFolds;
				int validationEnd = ds.nSamples*i / nFolds;
				svm->Train(validationStart, validationEnd, alpha1, b1);
				svm->Test(validationStart, validationEnd, alpha1, b1, correct);
				totalCorrect += correct;
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
			auto totalCorrect = 0;
			int correct;
			for (auto i = 1; i <= nFolds; i++){
				vector<double> alpha1;
				double b1;
				int validationStart = ds.nSamples*(i - 1) / nFolds;
				int validationEnd = ds.nSamples*i / nFolds;
				svm->Train(validationStart, validationEnd, alpha1, b1);
				svm->Test(validationStart, validationEnd, alpha1, b1, correct);
				totalCorrect += correct;
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
			auto totalCorrect = 0;
			int correct;
			for (auto i = 1; i <= nFolds; i++){
				vector<double> alpha1;
				double b1;
				int validationStart = ds.nSamples*(i - 1) / nFolds;
				int validationEnd = ds.nSamples*i / nFolds;
				svm->Train(validationStart, validationEnd, alpha1, b1);
				svm->Test(validationStart, validationEnd, alpha1, b1, correct);
				totalCorrect += correct;
			}
			double expected = true;
			double actual = 100.0*totalCorrect / ds.nSamples > 70.0;
			Assert::AreEqual(expected, actual);
		}

	};
}