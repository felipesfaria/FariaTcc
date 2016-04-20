#pragma once
#include "BaseSvm.h"
namespace FariaSvm{
	class SequentialSvm :
		public BaseSvm
	{
	public:
		SequentialSvm(shared_ptr<DataSet> ds);
		~SequentialSvm();
		int Classify(const TrainingSet& ts, const ValidationSet& vs, const int vIndex) override;
		void Train(TrainingSet& ts) override;
		double K(double* x, double* y, int size);
	private:
	};
}