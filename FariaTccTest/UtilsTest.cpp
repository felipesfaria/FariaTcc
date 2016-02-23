#include <stdio.h>
#include "stdafx.h"
#include "CppUnitTest.h"
#include "../FariaSvm/Utils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
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

		TEST_METHOD(Utils_GetComandVariable_FindVariable)
		{
			char *argv[3] = { "exe path", "-a", "b" };
			string expected = "b";
			auto actual = Utils::GetComandVariable(3, argv, "-a");
			Assert::AreEqual(actual, expected);
		}

	};
}