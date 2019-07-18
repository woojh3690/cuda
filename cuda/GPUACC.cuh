#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define __syncthreads()
#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif

	class GPUACC
	{
	public:
		GPUACC(void);
		virtual ~GPUACC(void);
		void MatrixMultiplication(float* M, float* N, float* P, int Width);
	};
	
#ifdef __cplusplus 
}
#endif