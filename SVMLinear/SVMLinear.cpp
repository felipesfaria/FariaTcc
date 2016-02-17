// SVMLinear.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <exception>
#include <ctime>
#include "Logger.h"
#include "Utils.h"
#include <algorithm>
#include "DataSet.h"

Logger logger;
class Kernel;
class SvmData;
using namespace std;

class DataSet2
{
public:
	string FileName;
	bool IsCsv;
	int nTrainers;
	int nTesters;
	int nFeatures;
	int nClasses;
	int C;
	double Gama;
	double Step;
	double Precision = 1e-9;
	vector<double> x;
	double y;

	string const& operator[](size_t index) const
	{
		return m_data[index];
	}
	size_t size() const
	{
		return m_data.size() + m_doubles.size() + m_longs.size();
	}
	void readNextRow(istream& str)
	{
		if (IsCsv)
			readCsvData(str);
		else
			readIndexedData(str);
	}
	DataSet2 static Get(string arg)
	{
		switch (arg[0])
		{
		case 'a':
			return GetAdult(arg[1]);
		case 'w':
			return GetWeb(arg[1]);
		case 'i':
		default:
			return GetIris();
		}
	}
private:
	vector<string> m_data;
	vector<double> m_doubles;
	vector<long> m_longs;
	string myClass;
	string classes[2];

	DataSet2 static GetAdult(char c)
	{
		return GetAdult(c - '0');
	}
	DataSet2 static GetAdult(int n)
	{
		switch (n)
		{
		case 1: return GetAdult1();
		case 2: return GetAdult2();
		case 3: return GetAdult3();
		case 4: return GetAdult4();
		case 5: return GetAdult5();
		case 6: return GetAdult6();
		case 7: return GetAdult7();
		case 8: return GetAdult8();
		case 9: return GetAdult9();
		default:
			throw(new exception("Invalid adult type"));
		}
	}
	DataSet2 static GetAdult1()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult1.data";
		ds.IsCsv = false;
		ds.nTrainers = 1605;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult2()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult2.data";
		ds.IsCsv = false;
		ds.nTrainers = 2265;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult3()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult3.data";
		ds.IsCsv = false;
		ds.nTrainers = 3185;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult4()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult4.data";
		ds.IsCsv = false;
		ds.nTrainers = 4781;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult5()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult5.data";
		ds.IsCsv = false;
		ds.nTrainers = 6414;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult6()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult6.data";
		ds.IsCsv = false;
		ds.nTrainers = 11220;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult7()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult7.data";
		ds.IsCsv = false;
		ds.nTrainers = 16100;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult8()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult8.data";
		ds.IsCsv = false;
		ds.nTrainers = 22696;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}
	DataSet2 static GetAdult9()
	{
		DataSet2 ds;
		ds.FileName = "Data/adult9.data";
		ds.IsCsv = false;
		ds.nTrainers = 32561;
		ds.nTesters = 16281;
		ds.nFeatures = 123;
		ds.nClasses = 2;
		ds.C = 100;
		ds.Gama = 0.5;
		ds.Step = 1e-12;
		return ds;
	}

	DataSet2 static GetWeb(char c)
	{
		return GetWeb(c - '0');
	}
	DataSet2 static GetWeb(int n)
	{
		switch (n)
		{
		case 1: return GetWeb1();
		case 2: return GetWeb2();
		case 3: return GetWeb3();
		case 4: return GetWeb4();
		case 5: return GetWeb5();
		case 6: return GetWeb6();
		case 7: return GetWeb7();
		case 8: return GetWeb8();
		default:
			throw(new exception("Invalid Web type"));
		}
	}
	DataSet2 static GetWeb1()
	{
		DataSet2 ds;
		ds.FileName = "Data/web1.data";
		ds.IsCsv = false;
		ds.nTrainers = 2477;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb2()
	{
		DataSet2 ds;
		ds.FileName = "Data/web2.data";
		ds.IsCsv = false;
		ds.nTrainers = 3470;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb3()
	{
		DataSet2 ds;
		ds.FileName = "Data/web3.data";
		ds.IsCsv = false;
		ds.nTrainers = 4912;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb4()
	{
		DataSet2 ds;
		ds.FileName = "Data/web4.data";
		ds.IsCsv = false;
		ds.nTrainers = 7366;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb5()
	{
		DataSet2 ds;
		ds.FileName = "Data/web5.data";
		ds.IsCsv = false;
		ds.nTrainers = 9888;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb6()
	{
		DataSet2 ds;
		ds.FileName = "Data/web6.data";
		ds.IsCsv = false;
		ds.nTrainers = 17188;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb7()
	{
		DataSet2 ds;
		ds.FileName = "Data/web7.data";
		ds.IsCsv = false;
		ds.nTrainers = 24692;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}
	DataSet2 static GetWeb8()
	{
		DataSet2 ds;
		ds.FileName = "Data/web8.data";
		ds.IsCsv = false;
		ds.nTrainers = 49749;
		ds.nTesters = 0;
		ds.nFeatures = 300;
		ds.nClasses = 2;
		ds.C = 64;
		ds.Gama = 7.8125;
		ds.Step = 1e-13;
		return ds;
	}

	DataSet2 static GetIris()
	{
		DataSet2 ds;
		ds.FileName = "Data/iris.csv";
		ds.IsCsv = true;
		ds.nTrainers = 80;
		ds.nTesters = 20;
		ds.nFeatures = 4;
		ds.nClasses = 2;
		ds.C = 16;
		ds.Gama = 0.5;
		ds.Step = 0.0001;
		return ds;
	}
	void readIndexedData(istream& str)
	{
		string         line;
		getline(str, line);

		if (line.size() == 0) return;
		stringstream   lineStream(line);
		string         cell;

		int i = 0;
		getline(lineStream, cell, ' ');
		//m_data.push_back(cell);
		if (!Utils::TryParseDouble(cell, y))
			throw new exception("Missing y value.");
		x.clear();
		for (int i = 0; i < nFeatures; ++i)
		{
			x.push_back(0);
		}

		while (getline(lineStream, cell, ' '))
		{
			//m_data.push_back(cell);
			auto elems = Utils::split(cell, ':');
			int index = stoi(elems[0]);
			x[index - 1] = stod(elems[1]);
		}
	}

	void readCsvData(istream& str)
	{
		string         line;
		getline(str, line);

		stringstream   lineStream(line);
		string         cell;

		m_data.clear();
		m_doubles.clear();
		m_longs.clear();
		x.clear();
		while (getline(lineStream, cell, ','))
		{
			m_data.push_back(cell);
			try
			{
				double t;
				t = stod(cell);
				m_doubles.push_back(t);
				x.push_back(t);
				continue;
			}
			catch (invalid_argument&)
			{

			}

			try
			{
				long t;
				t = stol(cell);
				m_longs.push_back(t);
				continue;
			}
			catch (invalid_argument&)
			{

			}
		}
		//Last line didn't read anything
		if (m_data.size() == 0) return;

		myClass = m_data[m_data.size() - 1];

		//Find out classes
		//static string SvmData::classes[2];
		if (classes[1].empty())
			if (classes[0].empty())
				classes[0] = myClass;
			else if (classes[0].compare(myClass) != 0)
				classes[1] = myClass;

		if (classes[0].compare(myClass) == 0)
			y = 1;
		else
			y = -1;
	}
};

istream& operator>>(istream& str, DataSet2& data)
{
	data.readNextRow(str);
	return str;
}

class Kernel
{
public:
	Kernel()
	{
		_type = NONE;
	}
	enum KernelType
	{
		NONE,
		LINEAR,
		HOMOGENEOUS_POLYNOMIAL,
		NONHOMOGENEOUS_POLYNOMIAL,
		GAUSSIAN

	};

	double K(vector<double> x, vector<double> y)
	{
		switch (_type)
		{
		case LINEAR:
			return Linear(x, y);
		case HOMOGENEOUS_POLYNOMIAL:
			return HomogeneousPolynomial(x, y);
		case NONHOMOGENEOUS_POLYNOMIAL:
			return NonHomogeneousPolynomial(x, y);
		case GAUSSIAN:
			return Gaussian(x, y);
		default:
			throw new exception("Kernel Type not defined or invalid type defined.");
		}
	}
	bool DefineLinear()
	{
		if (_type != NONE)
			throw new exception("Can't redefine Kernel");
		_type = LINEAR;
		return true;
	}
	double static Linear(vector<double> x, vector<double> y){
		if (x.size() != y.size())
			throw new exception("Incompatible sizes..");
		double sum = 0;
		for (int i = 0; i < x.size(); ++i)
			sum += x[i] * y[i];
		return sum;
	}

	bool DefineHomogeneousPolynomial(int d)
	{
		if (d < 2)
			throw new exception("Invalid argument d<2");
		if (_type != NONE)
			throw new exception("Can't redefine Kernel");
		_d = d;
		_type = HOMOGENEOUS_POLYNOMIAL;
		return true;
	}
	double HomogeneousPolynomial(vector<double> x, vector<double> y)
	{
		double linear = Linear(x, y);
		double product = 1;
		for (int i = 0; i < _d; ++i)
			product *= linear;
		return product;
	}

	bool DefineNonHomogeneousPolynomial(int d, double c)
	{
		if (d < 2)
			throw new exception("Invalid argument d<2");
		if (c <= 0)
			throw new exception("Invalid argument c<=0");
		if (_type != NONE)
			throw new exception("Can't redefine Kernel");
		_d = d;
		_c = c;
		_type = NONHOMOGENEOUS_POLYNOMIAL;
		return true;
	}
	double NonHomogeneousPolynomial(vector<double> x, vector<double> y)
	{
		double linear = Linear(x, y);
		double product = 1;
		for (int i = 0; i < _d; ++i)
			product *= linear + _c;
		return product;
	}

	bool DefineGaussian(double sigma)
	{
		if (_type != NONE)
			throw new exception("Can't redefine Kernel");
		_sigma = sigma;
		_type = GAUSSIAN;
		return true;
	}
	double Gaussian(vector<double> x, vector<double> y)
	{
		double sum = 0;
		double product;
		double gama = 1 / (2 * _sigma*_sigma);
		for (int i = 0; i < x.size(); ++i)
		{
			product = x[i] - y[i];
			product *= product;
			sum += product;
		}
		return exp(-gama*sum);
	}
	KernelType GetType()
	{
		return _type;
	}
private:
	KernelType _type;
	int _d;
	double _c;
	double _sigma;
};

int classify(DataSet& ds, int index, vector<double>& alpha, Kernel& kernel, double& b)
{
	auto precision = ds.Precision;
	auto x = ds.X;
	auto y = ds.Y;
	auto size = alpha.size();
	auto sum = 0.0;
	for (auto i = 0; i < alpha.size(); ++i)
	{
		if (alpha[i] == 0) continue;
		sum += alpha[i] * y[i] * kernel.K(x[i], x[index]);
	}
	auto sign = sum - b;
	if (sign > precision)
		return 1;
	if (sign < -precision)
		return -1;
	return 0;
}

void Train(DataSet& ds, int nTrainers, Kernel& kernel, vector<double>& alpha, double& b)
{
	Logger::FunctionStart("Train");
	alpha.clear();
	vector<double> oldAlpha;
	for (int i = 0; i < nTrainers; ++i){
		alpha.push_back(0);
		oldAlpha.push_back(1);
	}
	vector<vector<double>> x = ds.X;
	vector<double> y = ds.Y;
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = ds.Step;
	double C = ds.C;
	double precision = ds.Precision;
	do
	{
		count++;

		difAlpha = 0;
		for (int i = 0; i < nTrainers; ++i){
			difAlpha += alpha[i] - oldAlpha[i];
			oldAlpha[i] = alpha[i];
		}

		if (count>0)
			Logger::ClassifyProgress(count, step, lastDif, difAlpha);
			
		if (abs(difAlpha) < precision)
			break;
		if (abs(difAlpha - lastDif) > difAlpha / 10.0)
			step = step / 2;
		lastDif = difAlpha;
		for (int i = 0; i < nTrainers; ++i)
		{
			double sum = 0;
			for (int j = 0; j < nTrainers; ++j)
			{
				if (oldAlpha[j] == 0) continue;
				sum += y[j] * oldAlpha[j] * kernel.K(x[j], x[i]);
			}
			double value = oldAlpha[i] + step - step*y[i] * sum;
			if (value > C)
				alpha[i] = C;
			else if (value < 0)
				alpha[i] = 0.0;
			else
				alpha[i] = value;
		}

	} while (true);
	int nSupportVectors = 0;
	vector<double> sv;
	double maxValue = 0.0;
	for (auto i = 0; i < alpha.size(); ++i)
	{
		if (alpha[i] != 0) nSupportVectors++;
		if (alpha[i] > maxValue){
			maxValue = alpha[i];
			sv = x[i];
		}
	}
	if (maxValue == 0.0)
		throw new exception("Could not find support vector.");
	b = 0.0;
	Logger::Stats("nSupportVectors", nSupportVectors);
	Logger::FunctionEnd();
}

DataSet2 GetDataSet(int argc, char** argv)
{
	string arg = Utils::GetComandVariable(argc, argv, "-d");
	DataSet2 ds = DataSet2::Get(arg);

	Logger::Line("DataSet2:");
	Logger::Stats( "FileName: " ,ds.FileName );
	Logger::Stats( "Samples: " ,ds.nTrainers );
	Logger::Stats( "C: " ,ds.C );
	Logger::Stats( "Gama: " ,ds.Gama );
	Logger::Stats( "Precision: " ,ds.Precision );
	Logger::Stats( "Classes: " ,ds.nClasses );
	Logger::Stats( "Features: " ,ds.nFeatures );
	Logger::Stats( "InitialStepSize: " ,ds.Step );
	return ds;
}

void Test(DataSet& ds, int nValidators, int nTrainers, Kernel& kernel, vector<double>& alpha1, double& b1, int& nCorrect)
{
	Logger::FunctionStart("Test");
	nCorrect = 0;
	for (auto i = nTrainers; i < ds.X.size(); ++i)
	{
		int classifiedY = classify(ds, i, alpha1, kernel, b1);
		if (classifiedY == ds.Y[i]){
			nCorrect++;
		}
	}
	int end = clock();
	auto percentageCorrect = static_cast<double>(nCorrect) / nValidators;
	Logger::Percentage(nCorrect, nValidators, percentageCorrect);
	Logger::FunctionEnd();
}

void ReadFile(DataSet& ds)
{
	Logger::FunctionStart("ReadFile");
	ifstream       file;
	file.open(ds.FileName, ifstream::in);

	if (!file.good())
		throw(new exception("Error: File not found"));
	ds.ReadFile(file);
	Logger::FunctionEnd();
}

int main(int argc, char* argv[])
{
	try{
		unsigned seed = time(nullptr);
		unsigned int start = clock();
		Logger::Init(argc, argv);
		Logger::Seed(seed);
		srand(seed);

		DataSet ds(argc, argv);

		ReadFile(ds);

		Utils::Shuffle(ds.X, ds.Y);

		double validationPercentage = 0.5;
		int nValidators = (ds.X.size()*validationPercentage);
		int nTrainers = ds.X.size() - nValidators;
		Kernel kernel;
		kernel.DefineGaussian(ds.Gama);
		//kernel.DefineHomogeneousPolynomial(2);
		int iFold = 1;
		
		Logger::Fold(iFold++);

		vector<double> alpha1;
		double b1;
		int nCorrect1;
		Train(ds, nTrainers, kernel, alpha1, b1);
		Test(ds, nValidators, nTrainers, kernel, alpha1, b1, nCorrect1);

		Utils::Reverse(ds.X, ds.Y);

		Logger::Fold(iFold++);

		vector<double> alpha2;
		double b2;
		int nCorrect2;
		Train(ds, nTrainers, kernel, alpha2, b2);
		Test(ds, nValidators, nTrainers, kernel, alpha2, b2, nCorrect2);

		double totalCorrect = nCorrect1 + nCorrect2;
		double totalSamples = nValidators + nTrainers;
		double averagePercentageCorrect = totalCorrect / totalSamples;
		Logger::Percentage(totalCorrect, totalSamples, averagePercentageCorrect,"Average ");
		Logger::End();
		return 0;
	}
	catch (exception& e)
	{
		Logger::Error(e);
		return 1;
	}
}