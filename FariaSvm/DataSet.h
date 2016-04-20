#pragma once
#include "Enums.h"
#include "ValidationSet.h"
#include "TrainingSet.h"
#include <vector>

using namespace std;
namespace FariaSvm{
	class DataSet
	{
	public:
		string FileName;
		int nSamples;
		int nFeatures = 0;
		int nClasses = 2;
		unsigned nFolds;
		int C;
		double Gama = -1;
		vector<vector<double>> X;
		vector<double> Y;
		DataSet();
		~DataSet();
		void InitFoldSets(TrainingSet &ts, ValidationSet &vs, int fold);
		static string GetFileName(string arg);
	private:
		void InitData(string arg = "");
		void ReadFile();
		bool readRow(istream& str);
	};
}