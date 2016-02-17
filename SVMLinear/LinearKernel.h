#pragma once
#include <vector>

enum KernelType
{
	NONE,
	LINEAR,
	HOMOGENEOUS_POLYNOMIAL,
	NONHOMOGENEOUS_POLYNOMIAL,
	GAUSSIAN

};
class LinearKernel
{
public:
	LinearKernel();
	~LinearKernel();
	double K(std::vector<double> x, std::vector<double> y);
	bool DefineLinear();
	static double Linear(std::vector<double> x, std::vector<double> y);
	bool DefineHomogeneousPolynomial(int d);
	double HomogeneousPolynomial(std::vector<double> x, std::vector<double> y);
	bool DefineNonHomogeneousPolynomial(int d, double c);
	double NonHomogeneousPolynomial(std::vector<double> x, std::vector<double> y);
	bool DefineGaussian(double sigma);
	double Gaussian(std::vector<double> x, std::vector<double> y);
	KernelType GetType();
private:
	KernelType _type;
	int _d;
	double _c;
	double _sigma;
};

