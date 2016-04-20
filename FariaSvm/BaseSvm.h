#pragma once
#include "DataSet.h"
#include "TrainingSet.h"
#include "ValidationSet.h"
#include <memory>

namespace FariaSvm{
	class BaseSvm
	{
	public:
		BaseSvm(shared_ptr<DataSet> ds);
		static unique_ptr<BaseSvm> GenerateSvm(shared_ptr<DataSet> ds, string arg = "");
		virtual int Classify(const TrainingSet& ts, const ValidationSet& vs, const int vIndex);
		virtual void Train(TrainingSet & ts);
		virtual void Test(TrainingSet & ts, ValidationSet & vs);
		int SignOf(double value);
		virtual ~BaseSvm();
	protected:
		shared_ptr<DataSet> _ds;
		bool isMultiStep;
		bool isStochastic;
		double Precision;
		double g;
		double C;
		double _initialStep;
		unsigned MaxIterations;
	private:
	};
}
