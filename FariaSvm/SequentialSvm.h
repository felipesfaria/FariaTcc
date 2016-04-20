#pragma once
#include "BaseSvm.h"
namespace FariaSvm{
	class SequentialSvm :
		public BaseSvm
	{
	public:
		SequentialSvm(shared_ptr<DataSet> ds);
		~SequentialSvm();
		int Classify(TrainingSet& ts, ValidationSet& vs, int vIndex) override;
		void Train(TrainingSet & ts) override;
		double K(double* x, double* y, int size);
	private:
	};
}