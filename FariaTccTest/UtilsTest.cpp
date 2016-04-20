#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/Utils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace FariaSvm;
namespace FariaTccTest
{
	TEST_CLASS(UtilsTest)
	{
	public:
		TEST_METHOD(Utils_Shuffle_XandYSameShuffle)
		{
			srand(1);
			vector<vector<double>> a = { { 1.0}, {2.0}, {3.0},{4.0}};
			vector<double> b = { 2.0, 4.0, 6.0, 8.0 };
			Utils::Shuffle(a, b);
			for (int i = 0; i < 4; i++){
				Assert::AreEqual(a[i][0]*2, b[i]);
			}
		}

		TEST_METHOD(Utils_GetComandVariable_FindArgument)
		{
			char *argv[3] = { "exe path", "-a", "b" };
			string expected = "b";
			string actual;
			Utils::GetComandVariable(3, argv, "-a", actual);
			Assert::AreEqual(actual, expected);
		}

		TEST_METHOD(Utils_GetComandVariable_CommandExists)
		{
			char *argv[3] = { "exe path", "-a", "b" };
			string expected = "b";
			string arg;
			auto actual = Utils::GetComandVariable(3, argv, "-a", arg);
			Assert::IsTrue(actual);
		}

		TEST_METHOD(Utils_GetComandVariable_CommandDoesntExists)
		{
			char *argv[3] = { "exe path", "-a", "b" };
			string expected = "b";
			string arg;
			auto actual = Utils::GetComandVariable(3, argv, "-x", arg);
			Assert::IsFalse(actual);
		}

		TEST_METHOD(PadLeft)
		{
			string expected = "01";
			string actual = Utils::PadLeft("1", 2);
			Assert::AreEqual(expected, actual);
			expected = "002";
			actual = Utils::PadLeft("2", 3);
			Assert::AreEqual(expected, actual);
			expected = "003";
			actual = Utils::PadLeft(3, 3);
			Assert::AreEqual(expected, actual);
		}

	};
}