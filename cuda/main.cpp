#include <iostream>
#include <ctime>
#include "GPUACC.cuh"
#include <Windows.h>

void matrixPrint(float* data, int width);
float randomDouble(void);
void matrixCalulatorToCpu(float* Md, float* Nd, float* Pd, int Width);
void spec();

int main(int argc, char* argv[])
{
	//spec();
	
	std::cout << "Hello Cuda!" << std::endl;

	int Width = 20;
	int size = Width * Width * sizeof(float);
	float* aM = (float*)malloc(size);
	float* aN = (float*)malloc(size);
	float* aP = (float*)malloc(size);

	//초기화
	srand((unsigned int)time(NULL));
	for (int i = 0; i < Width * Width; i++)
	{
		aM[i] = i; //randomDouble();
		aN[i] = i;// randomDouble();
	}

	//출력
	std::cout << std::endl;
	matrixPrint(aM, Width);
	std::cout << " (*) " << std::endl;
	matrixPrint(aN, Width);
	std::cout << " = " << std::endl;



	clock_t start = clock();
	for (int i = 0; i < 1000; i++) {
		matrixCalulatorToCpu(aM, aN, aP, Width);
	}
	clock_t end = clock();
	matrixPrint(aP, Width);
	std::cout << "cpu 걸린 시간 : " << (double)end - start << std::endl;

	aP = (float*)malloc(size);

	GPUACC gpuacc;
	double time = gpuacc.MatrixMultiplication(aM, aN, aP, Width);
	matrixPrint(aP, Width);
	std::cout << "gpu 걸린 시간 : " << time << std::endl;
	
	return 0;
}

void matrixPrint(float* data, int width)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < width; j++)
		{
			std::cout << " " << data[i * width + j];
		}
		std::cout << std::endl;
	}
}

float randomDouble(void) {
	return (float)rand() / RAND_MAX;
}

void matrixCalulatorToCpu(float* Md, float* Nd, float* Pd, int Width)
{
	for (int i = 0; i < Width; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			float sum = 0;
			for (int k = 0; k < Width; k++)
			{
				float a = Md[i * Width + k];
				float b = Nd[k * Width + j];
				sum += a * b;
			}
			Pd[i * Width + j] = sum;
		}
	}
}

void spec()
{
	cudaDeviceProp prop;
	int count; cudaGetDeviceCount(&count);
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i); 
		printf("--- General Information for device %d ---\n", i); 
		printf("Name : %s\n", prop.name); 
		printf("Compute capability : %d.%d\n", prop.major, prop.minor);
		printf("clock rate : %d\n", prop.clockRate);
		printf("device copy overlap : "); 
		if (prop.deviceOverlap) 
			printf("enabled\n"); 
		else printf("Disabled\n"); 
		printf("Kernel execition timeout : "); 
		if (prop.kernelExecTimeoutEnabled) 
			printf("Enabled\n"); 
		else printf("Disabled\n");
		printf("--- Memory Information for device %d ---\n", i);
		printf("total global mem : %ld\n", prop.totalGlobalMem); 
		printf("Total constant mem : %ld\n", prop.totalConstMem);
		printf("Max mem pitch : %ld\n", prop.memPitch); 
		printf("Texture Alignment : %ld\n", prop.textureAlignment); 
		printf("--- MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); 
		printf("Shared mem per mp : %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp : %d\n", prop.regsPerBlock);
		printf("Threads in warp : %d\n", prop.warpSize);
		printf("Max threads per block : %d\n", prop.maxThreadsPerBlock); 
		printf("Max thread dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid imensions : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); 
		printf("\n");
	}

}