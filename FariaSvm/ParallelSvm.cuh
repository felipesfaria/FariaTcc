#pragma once
#include "BaseSvm.h"
#include <driver_types.h>
namespace FariaSvm{
	class CudaArray
	{
	public:
		double* device = nullptr;
		double* host = nullptr;
		int size = 0;
		bool deviceOnly = false;
		~CudaArray();
		void Init(double* host, int size);
		void Init(int size);
		void CopyToDevice() const;
		void CopyToHost() const;
		double GetSum() const;
	};

	class ParallelSvm :
		public BaseSvm
	{
	public:
		ParallelSvm(shared_ptr<DataSet> ds);
		~ParallelSvm();
		int Classify(TrainingSet& ts, ValidationSet& vs, int index) override;
		void UpdateBlocks(TrainingSet& ts);
		void Train(TrainingSet & ts) override;
		void PrepareTest(TrainingSet& ts, ValidationSet& vs);
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