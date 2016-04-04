#pragma once
#include <string>
#include <map>
using namespace std;
class Setting
{
public:
	enum Type
	{
		NONE,
		UNSIGNED,
		STRING,
		DOUBLE,
		HELP
	};
	string name;
	string command;
	Type type;
	string description;
	bool isSet;
	unsigned uValue;
	string sValue;
	double dValue;

	string ToString();
};
class Settings
{
public:
	map<string, Setting> settingsMap;
	static Settings *instance();
	void Init(int argc, char** argv);
	void GetUnsigned(string key, unsigned &value);
	string GetString(string key);
	void GetDouble(string key, double &value);
	void GetDouble(string key, double &value, double def);
	~Settings();
	void ShowHelp();
private:
	Settings();
	static Settings *s_instance;
};

