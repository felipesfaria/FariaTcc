
class Dados
{
public:
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
		string         line;
		getline(str, line);

		stringstream   lineStream(line);
		string         cell;

		m_data.clear();
		while (getline(lineStream, cell, ','))
		{
			m_data.push_back(cell);
			try
			{
				double t;
				t = stod(cell);
				m_doubles.push_back(t);
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
		static string classes[2];
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

	Dados Copy()
	{
		unsigned int i;
		Dados d;
		for (i = 0; i < m_data.size(); i++)
			d.m_data.push_back(m_data[i]);
		for (i = 0; i < m_doubles.size(); i++)
			d.m_doubles.push_back(m_doubles[i]);
		for (i = 0; i < m_longs.size(); i++)
			d.m_longs.push_back(m_longs[i]);
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
};