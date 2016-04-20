#pragma once
#include "DataSet.h"
#include "TrainingSet.h"
#include "ValidationSet.h"
#include <memory>

namespace FariaSvm{
	class BaseSvm
	{
	public:
		BaseSvm();
		BaseSvm(DataSet &ds);
		static unique_ptr<BaseSvm> GenerateSvm(DataSet& ds, string arg = "");
		virtual int Classify(TrainingSet& ts, ValidationSet& vs, int index);
		virtual void Train(TrainingSet & ts);
		virtual void Test(TrainingSet & ts, ValidationSet & vs);
		int SignOf(double value);
		virtual ~BaseSvm();
	protected:
		DataSet* _ds;
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
