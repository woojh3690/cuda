#include "GPUACC.cuh"

#define TILE_WIDTH 25

GPUACC::GPUACC(void)
{

}

GPUACC::~GPUACC(void)
{

}

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	//Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	//printf("Row : %d,  Col : %d\n", Row, Col);

	float Pvalue = 0;
	//Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width / TILE_WIDTH; m++)
	{
		//Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
		Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * Width + Col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}

	Pd[Row * Width + Col] = Pvalue;

	////2D Thread ID
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;

	////Pvalue stores the Pd element that is computed by the thread
	//float Pvalue = 0;

	//for (int k = 0; k < Width; k++)
	//{
	//	float Mdelement = Md[ty * Width + k];
	//	float Ndelement = Nd[k * Width + tx];
	//	//printf("ÁÂÇ¥ : %d, %d\n", ty * Width + k, k * Width + tx);
	//	Pvalue += Mdelement * Ndelement;
	//}

	////Write the matrix to device memory each thread writes one element
	//Pd[ty * Width + tx] = Pvalue;
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
	dim3 dimGrid(2, 2);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	//Launch the device computation threads!
	MatrixMulKernel <<<dimGrid,dimBlock>>> (Md, Nd, Pd, Width);


	//Transfer P from device to host
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//Free dvice matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd);
}