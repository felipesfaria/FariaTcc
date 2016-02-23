#pragma once
#include <vector>
#include "BaseKernel.h"

class SequentialKernel
	: public BaseKernel
{
public:
	~SequentialKernel();
	double K(int i, int j, const DataSet& ds) override;
	static double Linear(vector<double> x, vector<double> y);
	bool DefineHomogeneousPolynomial(int d);
	double HomogeneousPolynomial(vector<double> x, vector<double> y);
	bool DefineNonHomogeneousPolynomial(int d, double c);
	double NonHomogeneousPolynomial(vector<double> x, vector<double> y);
	double Gaussian(vector<double> x, vector<double> y);
	KernelType GetType();
private:
};

