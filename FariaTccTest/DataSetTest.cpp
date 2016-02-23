#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/DataSet.h"

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
		}

		TEST_METHOD(DataSet_ReadFile)
		{
			char *argv[3] = { "exe path", "-d", "a1" };
			DataSet ds(3, argv);
			int actual = ds.X.size();
			int notExpected = 0;
			Assert::AreNotEqual(notExpected, actual);
		}

	};
}