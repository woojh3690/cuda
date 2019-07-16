#include "GPUACC.cuh"
#include "cuda.h"
#include <cufft.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>

GPUACC::GPUACC(void)
{

}

GPUACC::~GPUACC(void)
{

}

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	//2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//Pvalue stores the Pd element that is computed by the thread
	float Pvalue = 0;

	for (int k = 0; k < Width; ++k)
	{
		float Mdelement = Md[ty * Width + k];
		float Ndelement = Nd[k * Width + tx];
		Pvalue += Mdelement * Ndelement;
	}

	//Write the matrix to device memory each thread writes one element
	Pd[ty * Width + tx] = Pvalue;
}

void GPUACC::MatrixMultiplication(float* M, float* N, float* P, int Width)
{

	int size = Width * Width * sizeof(float);
	float* Md;
	float* Nd;
	float* Pd;

	//Transfer M and N to device memory
	cudaMalloc((void**)& Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& Nd, size);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

	//Allocate P on the device
	cudaMalloc((void**)& Pd, size);

	//Kernel invocation code - to be shown later
	//Setup the executioin configuration
	dim3 dimBlock(Width, Width);
	dim3 dimGrid(1, 1);

	//Launch the device computation threads!
	MatrixMulKernel <<<dimBlock, dimGrid>>> (Md, Nd, Pd, Width);


	//Transfer P from device to host
	//cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//Free dvice matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd);
}