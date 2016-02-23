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
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i" };
			DataSet ds(argc, argv);
			BaseKernel *kernel = new SequentialKernel(ds);
			Assert::IsTrue(true);
			free(kernel);
		}
		TEST_METHOD(MemoKernel_Constructor)
		{
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i" };
			DataSet ds(argc, argv);
			BaseKernel *kernel = new MemoKernel(ds);
			Assert::IsTrue(true);
			free(kernel);
		}
		TEST_METHOD(ParallelKernel_Constructor)
		{
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i" };
			DataSet ds(argc, argv);
			BaseKernel *kernel = new ParallelKernel(ds);
			Assert::IsTrue(true);
			free(kernel);
		}

		TEST_METHOD(SequentialKernel_K_SequentialEqualsMemo)
		{
			srand(0);
			int argc = 3;
			char *argv[] = { "exePath", "-d", "i" };
			DataSet ds(argc, argv);
			BaseKernel *sequentialKernel = new SequentialKernel(ds);
			BaseKernel *memoKernel = new MemoKernel(ds);
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
			BaseKernel *sequentialKernel = new SequentialKernel(ds);
			BaseKernel *parallelKernel = new ParallelKernel(ds);
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