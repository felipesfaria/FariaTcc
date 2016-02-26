#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/DataSet.h"
#include "../FariaSvm/TrainingSet.h"
#include "../FariaSvm/ValidationSet.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
namespace FariaTccTest
{
	TEST_CLASS(UtilsTest)
	{
	public:
		TEST_METHOD(DataSet_Constructor_Iris)
		{
			char *args[] = {
				"i"
			};
			char *argv[] = { "exe path", "-d", "" };
			for (int i = 0; i < 1; i++){
				argv[2] = args[i];
				DataSet ds(3, argv);
			}
			Assert::IsTrue(true);
		}

		TEST_METHOD(DataSet_Constructor_Adult)
		{
			char *args[] = {
				"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"
			};
			char *argv[] = { "exe path", "-d", "" };
			for (int i = 0; i < 9; i++){
				argv[2] = args[i];
				DataSet ds(3, argv);
			}
			Assert::IsTrue(true);
		}

		TEST_METHOD(DataSet_Constructor_Web)
		{
			char *args[] = {
				"w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"
			};
			char *argv[] = { "exe path", "-d", "" };
			for (int i = 0; i < 8; i++){
				argv[2] = args[i];
				DataSet ds(3, argv);
			}
			Assert::IsTrue(true);
		}

		TEST_METHOD(DataSet_ReadFile)
		{
			char *argv[3] = { "exe path", "-d", "a1" };
			DataSet ds(3, argv);
			int actual = ds.X.size();
			int notExpected = 0;
			Assert::AreNotEqual(notExpected, actual);
		}

		TEST_METHOD(TrainingSet_Constructor)
		{
			TrainingSet ts;
			Assert::IsFalse(ts.initialised);
		}

		TEST_METHOD(TrainingSet_Init)
		{
			TrainingSet ts;
			int width = 4;
			int height = 4;
			ts.Init(height, width);

			int lastX = (height - 1)*width + (width - 1);
			int lastY = (height - 1);
			int lastAlpha = lastY;

			ts.x[lastX] = 0;
			ts.y[lastY] = 0;
			ts.alpha[lastAlpha] = 0;

			Assert::IsTrue(ts.initialised);
		}

		TEST_METHOD(ValidationSet_Init)
		{
			ValidationSet vs;
			int width = 4;
			int height = 4;
			vs.Init(height, width);

			int lastX = (height - 1)*width + (width - 1);
			int lastY = (height - 1);

			vs.x[lastX] = 0;
			vs.y[lastY] = 0;

			Assert::IsTrue(vs.initialised);
		}

		TEST_METHOD(ValidationSet_Constructor)
		{
			ValidationSet vs;
			Assert::IsFalse(vs.initialised);
		}

		TEST_METHOD(DataSet_InitFoldSets)
		{
			int argc = 3;
			char *argv[] = { "exe path", "-d", "i" };
			DataSet ds(argc, argv);
			TrainingSet ts;
			ValidationSet vs;
			for (auto i = 1; i <= ds.nFolds; i++){
				ds.InitFoldSets(&ts, &vs, i);
				Assert::IsTrue(ts.height + vs.height == ds.nSamples);
			}
		}
	};
}