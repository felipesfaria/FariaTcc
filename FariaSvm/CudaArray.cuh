#pragma once
#include <cuda_runtime_api.h>
#include "CudaKernels.cuh"
#include "Settings.h"

namespace FariaSvm{

	class CudaArray
	{
	public:
		double* device = nullptr;
		double* host = nullptr;
		int size = 0;
		bool deviceOnly = false;
		CudaArray(Settings settings);
		~CudaArray();
		void Init(double* host, int size);
		void Init(int size);
		void SetAll(double value) const;
		void CopyToDevice() const;
		void CopyToHost() const;
		double GetSum() const;
	private:
		unsigned _threadsPerBlock;
	};
}
