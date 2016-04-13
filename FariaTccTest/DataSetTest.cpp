#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/DataSet.h"
#include "../FariaSvm/TrainingSet.h"
#include "../FariaSvm/ValidationSet.h"
#include <Settings.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
namespace FariaTccTest
{
	TEST_CLASS(DataSetTest)
	{
	public:
		TEST_METHOD(DataSet_ReadFile)
		{
			DataSet ds;
			int actual = ds.X.size();
			int expected = 100;
			Assert::AreEqual(expected, actual);
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
			vector<char*> argv =
			{
				"exePath",
				"-d", "i",
				"-f", "3",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;
			TrainingSet ts;
			ValidationSet vs;
			for (auto i = 1; i <= ds.nFolds; i++){
				ds.InitFoldSets(&ts, &vs, i);
				Assert::IsTrue(ts.height + vs.height == ds.nSamples);
			}
		}

		TEST_METHOD(DataReader_GetFileName)
		{
			string actual = DataSet::GetFileName("a1");
			string expected = ("Data/adult1.data");
			Assert::AreEqual(expected, actual);

			actual = DataSet::GetFileName("i");
			expected = ("Data/iris.data");
			Assert::AreEqual(expected, actual);

			actual = DataSet::GetFileName("w1");
			expected = ("Data/web1.data");
			Assert::AreEqual(expected, actual);
		}

		TEST_METHOD(DataSet_TestLoko)
		{
			vector<char*> argv =
			{
				"exePath",
				"-d", "a1",
				"-l", "n"
			};
			Settings::instance()->Init(argv.size(), argv.data());
			DataSet ds;
			int actual = ds.nSamples;
			int expected = 1605;
			Assert::AreEqual(expected, actual);

			actual = ds.nFeatures;
			expected = 119;
			Assert::AreEqual(expected, actual);
		}
	};
}