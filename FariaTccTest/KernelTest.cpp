#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/SequentialKernel.h"
#include "../FariaSvm/ParallelKernel.cuh"
#include <MemoKernel.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
namespace FariaTccTest
{
	TEST_CLASS(KernelTest)
	{
	public:
		TEST_METHOD(SequentialKernel_Constructor)
		{
			BaseKernel *kernel = new SequentialKernel;
			Assert::IsTrue(true);
			free(kernel);
		}
		TEST_METHOD(MemoKernel_Constructor)
		{
			BaseKernel *kernel = new MemoKernel;
			Assert::IsTrue(true);
			free(kernel);
		}
		TEST_METHOD(ParallelKernel_Constructor)
		{
			BaseKernel *kernel = new ParallelKernel;
			Assert::IsTrue(true);
			free(kernel);
		}

		TEST_METHOD(SequentialKernel_K_SequentialEqualsMemo)
		{
			srand(0);
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i" };
			DataSet ds(argc, argv);
			BaseKernel *sequentialKernel = new SequentialKernel;
			sequentialKernel->Init(ds);
			BaseKernel *memoKernel = new MemoKernel;
			memoKernel->Init(ds);
			for (int i = 0; i < ds.nSamples; i++)
				for (int j = 0; j < i; j++)
				{
					auto expected = sequentialKernel->K(i, j, ds);
					auto actual = memoKernel->K(i, j, ds);
					Assert::AreEqual(expected, actual);
				}
		}

		TEST_METHOD(SequentialKernel_K_SequentialEqualsParallel)
		{
			srand(0);
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i" };
			DataSet ds(argc, argv);
			BaseKernel *sequentialKernel = new SequentialKernel;
			sequentialKernel->Init(ds);
			BaseKernel *parallelKernel = new ParallelKernel;
			parallelKernel->Init(ds);
			for (int i = 0; i < ds.nSamples; i++)
				for (int j = 0; j < i; j++)
				{
					auto expected = sequentialKernel->K(i, j, ds);
					auto actual = parallelKernel->K(i, j, ds);
					Assert::AreEqual(expected, actual);
				}
		}

	};
}