#include "GPUACC.cuh"

#define TILE_WIDTH 7

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
}

double GPUACC::MatrixMultiplication(float* M, float* N, float* P, int Width)
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
	dim3 dimGrid(3, 3);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	//Launch the device computation threads!
	clock_t start = clock();
	for (int i = 0; i < 1000; i++) {
		MatrixMulKernel <<<dimGrid, dimBlock >>> (Md, Nd, Pd, Width);
	}
	clock_t end = clock();

	//Transfer P from device to host
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//Free dvice matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd);

	return (double)end - start;
}