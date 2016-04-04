#include "stdafx.h"
#include "Utils.h"
#include <sstream>


Utils::Utils()
{
}

Utils::~Utils()
{
}

bool Utils::TryParseDouble(std::string str, double &out)
{
	try
	{
		out = std::stod(str);
		return true;
	}
	catch (std::invalid_argument&)
	{
		return false;
	}
}

bool Utils::TryParseInt(std::string str, int& out)
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

bool Utils::TryParseInt(std::string str, unsigned& out)
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

vector<string>& Utils::split(const string& s, char delim, vector<string>& elems)
{
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> Utils::split(const std::string& s, char delim)
{
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

string Utils::GetComandVariable(int argc, char** argv, string comand)
{
	string argument = "";
	if (argc > 2){
		for (int i = 0; i < argc - 1; i++)
		{
			if (argv[i] == comand)
			{
				argument = argv[i + 1];
			}
		}
	}
	return argument;
}

void Utils::Shuffle(vector<vector<double>> &x, vector<double> &y)
{
	int size = x.size();
	for (auto i = 0; i < size; ++i)
	{
		auto position = i + rand() % (size - i);
		auto tx = x[position];
		auto ty = y[position];
		x[position] = x[i];
		y[position] = y[i];
		x[i] = tx;
		y[i] = ty;
	}
}

void Utils::Reverse(vector<vector<double>> &x, vector<double> &y)
{
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
}