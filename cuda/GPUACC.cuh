#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
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