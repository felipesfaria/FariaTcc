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
	double Precision = 1e-9;
	vector<vector<double>> X;
	vector<double> Y;
	std::string const& operator[](size_t index) const;
	size_t size() const;
	DataSet(int argc,char** argv);
	~DataSet();
	void ReadFile(istream& str);
	bool readNextRow(istream& str);
	void Init(string arg="");
private:
	vector<double> m_doubles;
	vector<long> m_longs;
	string classes[2];

	void InitAdult(char c);
	void InitAdult(int n);
	void InitWeb(char c);
	void InitWeb(int n);
	void InitIris();

	void readIndexedData(istream& str);
	void readCsvData(istream& str);
};

