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

class Kernel;
class SvmData;
using namespace std;
class DataSet
{
public:
	string FileName;
	bool IsCsv;
	int nTrainers;
	int nTesters;
	int nFeatures;
	int nClasses;
	DataSet static GetIris()
	{
		DataSet ds;
		ds.FileName = "../data/iris.csv";
		ds.IsCsv = true;
		ds.nTrainers = 80;
		ds.nTesters = 20;
		ds.nFeatures = 4;
		ds.nClasses = 2;
		return ds;
	}
};
class SvmData
{
public:
	vector<double> x;
	double y;
	static int nFeatures;
	static bool IsCsv;

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

	SvmData Copy()
	{
		unsigned int i;
		SvmData d;
		for (i = 0; i < m_data.size(); i++)
			d.m_data.push_back(m_data[i]);
		for (i = 0; i < m_doubles.size(); i++)
			d.m_doubles.push_back(m_doubles[i]);
		for (i = 0; i < m_longs.size(); i++)
			d.m_longs.push_back(m_longs[i]);
		for (i = 0; i < x.size(); i++)
			d.x.push_back(x[i]);
		d.myClass = myClass;
		d.y = y;

		return d;
	}
	string ToString()
	{
		unsigned int i;
		string output = "";
		for (i = 0; i < m_data.size(); i++)
		{
			output += m_data[i];
			if (i != m_data.size() - 1)
				output += ",";
		}
		output += "\n";
		return output;
	}
private:
	vector<string> m_data;
	vector<double> m_doubles;
	vector<long> m_longs;
	vector<string> m_multivalues;
	string myClass;
	static string classes[2];

	void readIndexedData(istream& str)
	{
		int i = 0;
		for (int i = 0; i < nFeatures; ++i)
		{
			
		}
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
		//static string classes[2];
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

istream& operator>>(istream& str, SvmData& data)
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

class SequentialSVM
{
public:
	SequentialSVM()
	{

	}
	void SetData(vector<SvmData> data)
	{
		_data = data;
	}
	void SetKernel(Kernel k)
	{
		_kernel = k;
	}
	void SetStep(double step)
	{
		_step = step;
	}
	void Init()
	{
		if (_data.size()==0)
			throw new exception("Tried to initialise without data.");
	}
private:
	vector<SvmData> _data;
	Kernel _kernel;
	double _step;
};

void shuffle(vector<SvmData> &dados)
{
	int size = dados.size();
	for (auto i = 0; i < size; ++i)
	{
		auto position = rand() % (size - i);
		auto t = dados[position];
		dados[position] = dados[i];
		dados[i] = t;
	}
}

int classify(vector<SvmData> data, SvmData example, vector<double> alpha, Kernel kernel)
{
	auto b = 0.0;
	auto size = alpha.size();
	vector<int> alpha_indexes;
	for (auto i = 0; i < alpha.size(); ++i)
	{
		auto value = alpha[i];
		if (value > 1e-6)
			alpha_indexes.push_back(i);
	}
	auto sv = data[alpha_indexes[0]];
	for (auto i = 0; i < alpha_indexes.size(); ++i)
	{
		auto index = alpha_indexes[i];
		b += alpha[index] * data[index].y*kernel.K(data[index].x, sv.x);
	}
	auto sum = 0.0;
	for (auto i = 0; i < alpha_indexes.size(); ++i)
	{
		auto index = alpha_indexes[i];
		sum += alpha[index] * data[index].y * kernel.K(data[index].x, sv.x);
	}
	if (sum > 1e-6)
		return 1;
	if (sum < -1e-6)
		return -1;
	return 0;
}

int main()
{
	srand(time(nullptr));
	DataSet ds = DataSet::GetIris();
	SvmData::IsCsv = ds.IsCsv;
	if (ds.IsCsv){
		SvmData::nFeatures = ds.IsCsv;
	}
	ifstream       file(ds.FileName);

	if (!file.good())
	{
		cout << "Error: File not found" << endl;
		return 1;
	}
	SvmData linha;
	vector<SvmData> linhas;
	SequentialSVM svm;
	double value, n, C, b, sum;
	n = 0.0001;
	C = 50.0;
	b = 0.0;

	while (file >> linha)
	{
		linhas.push_back(linha.Copy());
		cout << linha.ToString();
	}

	svm.SetData(linhas);

	shuffle(linhas);

	double validationPercentage = 0.2;
	int nValidators = (linhas.size()*validationPercentage);
	int nTrainers = linhas.size() - nValidators;
	Kernel kernel;
	kernel.DefineHomogeneousPolynomial(2);
	vector<double> alpha, oldAlpha;
	for (int i = 0; i < nTrainers; ++i){
		alpha.push_back(0);
		oldAlpha.push_back(1);
	}
	int count = 0;
	do
	{
		count++;
		double difAlpha = 0;
		for (int i = 0; i < nTrainers; ++i){
			difAlpha += alpha[i] - oldAlpha[i];
			oldAlpha[i] = alpha[i];
		}

		if (difAlpha<1e-4 && difAlpha>-1e-4)
			break;

		for (int i = 0; i < nTrainers; ++i)
		{
			sum = 0;
			for (int j = 0; j < nTrainers; ++j)
			{
				sum += linhas[j].y*alpha[j] * kernel.K(linhas[j].x, linhas[i].x);
			}
			value = alpha[i] + n - n*linhas[i].y*sum;
			if (value > C)
				alpha[i] = C;
			else if (value < 0)
				alpha[i] = 0.0;
			else
				alpha[i] = value;
		}

	} while (true);
	cout << "iterations: " << count << endl;
	auto nCorrect = 0;
	for (auto i = nTrainers; i < linhas.size(); ++i)
	{
		int classifiedY = classify(linhas, linhas[i], alpha, kernel);
		if (classifiedY == linhas[i].y){
			nCorrect++;
			cout << "Correct";
		}
		else
			cout << "Wrong";
	}
	auto percentageCorrect = static_cast<double>(nCorrect) / nValidators;
	cout << "Percentage correct: " << percentageCorrect*100.0 << "\n";
	return 0;
}