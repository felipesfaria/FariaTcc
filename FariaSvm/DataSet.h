#pragma once
#include <string>
#include <vector>
#include "Enums.h"

using namespace std;

class DataSet
{
public:
	string FileName;
	bool IsCsv;
	int nSamples;
	int nTesters;
	int nFeatures;
	int nClasses;
	int C;
	KernelType kernelType;
	double Gama;
	double Step;
	double Precision;
	int nFolds;
	int nTrainingSize;
	vector<vector<double>> X;
	vector<double> Y;
	std::string const& operator[](size_t index) const;
	size_t size() const;
	DataSet(int argc,char** argv);
	~DataSet();
	void InitData(string arg="");
private:
	vector<double> m_doubles;
	vector<long> m_longs;
	string classes[2];

	void ReadFile();
	bool readNextRow(istream& str);

	void InitAdult(char c);
	void InitAdult(int n);
	void InitWeb(char c);
	void InitWeb(int n);
	void InitIris();

	void readIndexedData(istream& str);
	void readCsvData(istream& str);
};

