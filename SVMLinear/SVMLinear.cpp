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
class Utils
{
public:
	bool static TryParseDouble(string str, double &out)
	{
		try
		{
			out = stod(str);
			return true;
		}
		catch (invalid_argument&)
		{
			return false;
		}
	}

	bool static TryParseInt(string str, int &out)
	{
		try
		{
			out = stoi(str);
			return true;
		}
		catch (invalid_argument&)
		{
			return false;
		}
	}

	std::vector<std::string> static &split(const std::string &s, char delim, std::vector<std::string> &elems) {
		std::stringstream ss(s);
		std::string item;
		while (std::getline(ss, item, delim)) {
			elems.push_back(item);
		}
		return elems;
	}

	std::vector<std::string> static split(const std::string &s, char delim) {
		std::vector<std::string> elems;
		split(s, delim, elems);
		return elems;
	}
};

string FormatClock(int milliseconds)
{
	stringstream ss;
	auto hours = milliseconds / (60 * 60 * CLOCKS_PER_SEC);
	milliseconds = milliseconds % (60 * 60 * CLOCKS_PER_SEC);
	auto minutes = milliseconds / (60 * CLOCKS_PER_SEC);
	milliseconds = milliseconds % (60 * CLOCKS_PER_SEC);
	auto seconds = milliseconds / (CLOCKS_PER_SEC);
	milliseconds = milliseconds % (CLOCKS_PER_SEC);
	ss << hours << ":" << minutes << ":" << seconds << ":" << milliseconds;
	return ss.str();
}

string FormatClock()
{
	return FormatClock(clock());
}

class DataSet
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
	DataSet static GetAdult1()
	{
		DataSet ds;
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
	DataSet static GetAdult2()
	{
		DataSet ds;
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
	DataSet static GetAdult3()
	{
		DataSet ds;
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
	DataSet static GetAdult4()
	{
		DataSet ds;
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
	DataSet static GetAdult5()
	{
		DataSet ds;
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
	DataSet static GetAdult6()
	{
		DataSet ds;
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
	DataSet static GetAdult7()
	{
		DataSet ds;
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
	DataSet static GetAdult8()
	{
		DataSet ds;
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
	DataSet static GetAdult9()
	{
		DataSet ds;
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
	DataSet static GetWeb1()
	{
		DataSet ds;
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
	DataSet static GetWeb2()
	{
		DataSet ds;
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
	DataSet static GetWeb3()
	{
		DataSet ds;
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
	DataSet static GetWeb4()
	{
		DataSet ds;
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
	DataSet static GetWeb5()
	{
		DataSet ds;
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
	DataSet static GetWeb6()
	{
		DataSet ds;
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
	DataSet static GetWeb7()
	{
		DataSet ds;
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
	DataSet static GetWeb8()
	{
		DataSet ds;
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
	DataSet static GetIris()
	{
		DataSet ds;
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
		if (SvmData::IsCsv)
			readCsvData(str);
		else
			readIndexedData(str);
	}

	SvmData Copy()
	{

		unsigned int i;
		SvmData d;
		//for (i = 0; i < m_data.size(); i++)
		//	d.m_data.push_back(m_data[i]);
		//for (i = 0; i < m_doubles.size(); i++)
		//	d.m_doubles.push_back(m_doubles[i]);
		//for (i = 0; i < m_longs.size(); i++)
		//	d.m_longs.push_back(m_longs[i]);
		for (i = 0; i < x.size(); i++)
			d.x.push_back(x[i]);
		//d.myClass = myClass;
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
		if (SvmData::classes[1].empty())
			if (SvmData::classes[0].empty())
				SvmData::classes[0] = myClass;
			else if (SvmData::classes[0].compare(myClass) != 0)
				SvmData::classes[1] = myClass;

		if (SvmData::classes[0].compare(myClass) == 0)
			y = 1;
		else
			y = -1;
	}
};
int SvmData::nFeatures = 0;
bool SvmData::IsCsv = false;
string SvmData::classes[] = { "", "" };

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
		if (_data.size() == 0)
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

void shuffle(vector<vector<double>> &x, vector<double> &y)
{
	cout << FormatClock() << "Shuffle starting..." << endl;
	int start = clock();
	int size = x.size();
	for (auto i = 0; i < size; ++i)
	{
		auto position = rand() % (size - i);
		auto tx = x[position];
		auto ty = y[position];
		x[position] = x[i];
		y[position] = y[i];
		x[i] = tx;
		y[i] = ty;
	}
	int end = clock();
	cout << FormatClock() << "Shuffle finished in "<<FormatClock(end-start) << endl;
}

void reverse(vector<vector<double>> &x, vector<double> &y)
{
	cout << FormatClock() << "Reverse starting..." << endl;
	int start = clock();
	int size = x.size();
	for (auto i = 0; i < size / 2; ++i)
	{
		auto position = size - i - 1;
		auto tx = x[position];
		auto ty = y[position];
		x[position] = x[i];
		y[position] = y[i];
		x[i] = tx;
		y[i] = ty;
	}
	int end = clock();
	cout << FormatClock() << "Reverse finished in " << FormatClock(end - start) << endl;
}

int classify(vector<vector<double>>& x, vector<double>& y, int index, vector<double>& alpha, Kernel& kernel, int precision, double& b)
{
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

//int classify(vector<SvmData> data, SvmData example, vector<double> alpha, Kernel kernel)
//{
//	return classify(data, example.x, example.y, alpha, kernel);
//}
void log(string msg)
{
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

void Train(vector<vector<double>>& x, vector<double>& y, DataSet ds, int nTrainers, Kernel& kernel, vector<double>& alpha,double& b)
{
	int start = clock();
	cout << FormatClock() << "\t Training Started" << endl;;
	alpha.clear();
	vector<double> oldAlpha;
	for (int i = 0; i < nTrainers; ++i){
		alpha.push_back(0);
		oldAlpha.push_back(1);
	}
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
			cout << FormatClock() << "\t Iteration: " << count << "\tstep: " << step << "\tlastDif:" << lastDif << "\tdifAlpha:" << difAlpha << endl;
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
	b = 0.0;
	vector<double> sv;
	double maxValue = 0.0;
	for (auto i = 0; i < alpha.size(); ++i)
	{
		if (alpha[i] > maxValue){
			maxValue = alpha[i];
			sv = x[i];
		}
	}
	if (maxValue == 0.0)
		throw new exception("Could not find support vector.");
	for (auto i = 0; i < alpha.size(); i++)
	{
		if (alpha[i] == 0) continue;
		b += alpha[i] * y[i] * kernel.K(x[i], sv);
	}
	int end = clock();
	cout << FormatClock() << "\t Training Finished in "<<FormatClock(end-start) << endl;
}

DataSet GetDataSet(int argc, char** argv)
{
	DataSet ds;
	char * datasetArg = getCmdOption(argv, argv + argc, "-d");
	if (cmdOptionExists(argv, argv + argc, "a1")){
		ds = DataSet::GetAdult1();
	}
	else if (cmdOptionExists(argv, argv + argc, "a2")){
		ds = DataSet::GetAdult2();
	}
	else if (cmdOptionExists(argv, argv + argc, "a3"))
	{
		ds = DataSet::GetAdult3();
	}
	else if (cmdOptionExists(argv, argv + argc, "a4"))
	{
		ds = DataSet::GetAdult4();
	}
	else if (cmdOptionExists(argv, argv + argc, "a5"))
	{
		ds = DataSet::GetAdult5();
	}
	else if (cmdOptionExists(argv, argv + argc, "a6"))
	{
		ds = DataSet::GetAdult6();
	}
	else if (cmdOptionExists(argv, argv + argc, "a7"))
	{
		ds = DataSet::GetAdult7();
	}
	else if (cmdOptionExists(argv, argv + argc, "a8"))
	{
		ds = DataSet::GetAdult8();
	}
	else if (cmdOptionExists(argv, argv + argc, "a9"))
	{
		ds = DataSet::GetAdult9();
	}
	else if (cmdOptionExists(argv, argv + argc, "w1")){
		ds = DataSet::GetWeb1();
	}
	else if (cmdOptionExists(argv, argv + argc, "w2")){
		ds = DataSet::GetWeb2();
	}
	else if (cmdOptionExists(argv, argv + argc, "w3"))
	{
		ds = DataSet::GetWeb3();
	}
	else if (cmdOptionExists(argv, argv + argc, "w4"))
	{
		ds = DataSet::GetWeb4();
	}
	else if (cmdOptionExists(argv, argv + argc, "w5"))
	{
		ds = DataSet::GetWeb5();
	}
	else if (cmdOptionExists(argv, argv + argc, "w6"))
	{
		ds = DataSet::GetWeb6();
	}
	else if (cmdOptionExists(argv, argv + argc, "w7"))
	{
		ds = DataSet::GetWeb7();
	}
	else if (cmdOptionExists(argv, argv + argc, "w8"))
	{
		ds = DataSet::GetWeb8();
	}
	else
	{
		ds = DataSet::GetIris();
	}
	cout << FormatClock() << "DataSet:" << endl;
	cout << FormatClock() << "FileName: " << ds.FileName << endl;
	cout << FormatClock() << "Samples: " << ds.nTrainers << endl;
	cout << FormatClock() << "C: " << ds.C << endl;
	cout << FormatClock() << "Gama: " << ds.Gama << endl;
	cout << FormatClock() << "Precision: " << ds.Precision << endl;
	cout << FormatClock() << "Classes: " << ds.nClasses << endl;
	cout << FormatClock() << "Features: " << ds.nFeatures << endl;
	cout << FormatClock() << "InitialStepSize: " << ds.Step << endl;
	return ds;
}

void Test(vector<vector<double>>& x, vector<double>& y, DataSet& ds, int nValidators, int nTrainers, Kernel& kernel, vector<double>& alpha1, double& b1, int& nCorrect)
{
	int start = clock();
	cout << FormatClock() << "\t Test Started" << endl;
	nCorrect = 0;
	for (auto i = nTrainers; i < x.size(); ++i)
	{
		int classifiedY = classify(x, y, i, alpha1, kernel, ds.Precision,b1);
		if (classifiedY == y[i]){
			nCorrect++;
		}
	}
	int end = clock();
	auto percentageCorrect = static_cast<double>(nCorrect) / nValidators;
	std::cout << FormatClock() << ": Percentage correct: " <<nCorrect<<"/"<<nValidators<<" = "<< percentageCorrect*100.0 << "%" << endl;
	cout << FormatClock() << "\t Test Finished in " << FormatClock(end - start) << endl;
}

void ReadFile(vector<vector<double>>& x, vector<double>& y, DataSet ds)
{
	cout << FormatClock() << ": Started reading file" << endl;
	int start = clock();
	ifstream       file;
	//ds.FileName = "../" + ds.FileName;
	file.open(ds.FileName, ifstream::in);

	if (!file.good())
	{
		std::cout << "Error: File not found" << endl;
	}
	SvmData linha;
	vector<SvmData> linhas;
	SequentialSVM svm;

	while (file >> linha)
	{
		//linhas.push_back(linha.Copy());
		vector<double> vd;
		for (int i = 0; i < linha.x.size(); ++i)
			vd.push_back(linha.x[i]);
		x.push_back(vd);
		y.push_back(linha.y);
		//std::cout << linha.ToString();
	}
	svm.SetData(linhas);
	int end = clock();
	cout << FormatClock() << ": Finished reading file in "<<FormatClock(end-start)<< endl;
}

int main(int argc, char* argv[])
{
	//unsigned seed = time(nullptr);
	unsigned seed = time(nullptr);
	unsigned int start = clock();
	cout << FormatClock() << "\t Program Started" << endl;;
	std::cout << FormatClock() << ": seed: " << seed << endl;
	srand(seed);
	vector<vector<double>> x;
	vector<double> y;
	DataSet ds = GetDataSet(argc, argv);
	
	//DataSet ds = DataSet::GetIris();
	SvmData::IsCsv = ds.IsCsv;
	SvmData::nFeatures = ds.nFeatures;

	ReadFile(x, y, ds);

	shuffle(x, y);

	double validationPercentage = 0.5;
	int nValidators = (x.size()*validationPercentage);
	int nTrainers = x.size() - nValidators;
	Kernel kernel;
	kernel.DefineGaussian(ds.Gama);
	//kernel.DefineHomogeneousPolynomial(2);

	cout << FormatClock() << "\tFirst Fold: " << endl;
	vector<double> alpha1;
	double b1;
	int nCorrect1;
	Train(x, y, ds, nTrainers, kernel,alpha1,b1);
	Test(x, y, ds, nValidators, nTrainers, kernel, alpha1, b1, nCorrect1);

	reverse(x, y);

	cout << FormatClock() << "\tSecond Fold: " << endl;
	vector<double> alpha2;
	double b2;
	int nCorrect2;
	Train(x, y, ds, nTrainers, kernel, alpha2, b2);
	Test(x, y, ds, nValidators, nTrainers, kernel, alpha2, b2, nCorrect2);

	double totalCorrect = nCorrect1 + nCorrect2;
	double totalSamples = nValidators + nTrainers;
	double averagePercentageCorrect = totalCorrect / totalSamples;
	std::cout << FormatClock() << ": Percentage correct: " << totalCorrect << "/" << totalSamples << " = " << averagePercentageCorrect*100.0 << "%" << endl;
	int end = clock();
	cout << FormatClock() << "\t Program Finished in " << FormatClock(end - start) << endl;
	std::cout << endl;
	return 0;
}