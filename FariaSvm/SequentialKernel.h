#pragma once
#include <vector>
#include "SvmKernel.h"

class SequentialKernel: public BaseKernel
{
public:
	~SequentialKernel();
	double K(std::vector<double> x, std::vector<double> y) override;
	bool DefineLinear();
	static double Linear(std::vector<double> x, std::vector<double> y);
	bool DefineHomogeneousPolynomial(int d);
	double HomogeneousPolynomial(std::vector<double> x, std::vector<double> y);
	bool DefineNonHomogeneousPolynomial(int d, double c);
	double NonHomogeneousPolynomial(std::vector<double> x, std::vector<double> y);
	double Gaussian(std::vector<double> x, std::vector<double> y) override;
	KernelType GetType();
private:
};

