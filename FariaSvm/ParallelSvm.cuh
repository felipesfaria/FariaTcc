#pragma once
#include "BaseSvm.h"
#include "CudaArray.cuh"
#include <driver_types.h>
namespace FariaSvm{
	class ParallelSvm :
		public BaseSvm
	{
	public:
		ParallelSvm(shared_ptr<DataSet> ds);
		~ParallelSvm();
		int Classify(const TrainingSet& ts, const ValidationSet& vs, const int vIndex) override;
		void UpdateBlocks(TrainingSet& ts);
		void Train(TrainingSet & ts) override;
		void PrepareTest(const TrainingSet& ts, const ValidationSet& vs);
	private:
		int _blocks = 1;
		unsigned _threadsPerBlock = 128;
		bool isTestPrepared = false;
		CudaArray caTrainingX;
		CudaArray caTrainingY;
		CudaArray caValidationX;
		CudaArray caAlpha;
		CudaArray caSum;
		CudaArray caStep;
		CudaArray caLastDif;
	};
}