#pragma once
#include <string>
#include <vector>
#include "Logger.h"
using namespace std;
class Utils
{
public:
	Utils();
	~Utils();
	bool static TryParseDouble(std::string str, double &out);
	bool static TryParseInt(std::string str, int &out);
	bool static TryParseInt(std::string str, unsigned &out);
	vector<string> static &split(const string &s, char delim, vector<string> &elems);
	std::vector<std::string> static split(const std::string &s, char delim);
	bool static GetComandVariable(int argc, char** argv, string comand, string &arg);
	void static Shuffle(vector<vector<double>>& x, vector<double>& y);
	void static Reverse(vector<vector<double>>& x, vector<double>& y);
};