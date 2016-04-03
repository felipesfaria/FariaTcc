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
	string key;
	Type type;
	string description;
	bool isSet;
	void SetValue(void* value);

	unsigned uValue;
	void GetUnsigned(unsigned &value);

	string sValue;
	void GetString(string &value);

	double dValue;
	void GetDouble(double &value);

	string ToString();
};
class Settings
{
public:
	map<string, Setting> settingsMap;
	static Settings *instance();
	void Init(int argc, char** argv);
	void GetUnsigned(string key, unsigned &value);
	void GetString(string key, string &value);
	void GetDouble(string key, double &value);
	void GetDouble(string key, double &value, double def);
	~Settings();
	void ShowHelp();
private:
	Settings();
	static Settings *s_instance;
};

