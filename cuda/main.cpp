#include <iostream>
#include "GPUACC.cuh"

void matrixPrint(float* data, int width);

int main(int argc, char* argv[])
{
	std::cout << "hello world" << std::endl;

	int Width = 20;
	int size = Width * Width * sizeof(float);
	float* aM = (float*)malloc(size);
	float* aN = (float*)malloc(size);
	float* aP = (float*)malloc(size);

	//초기화
	for (int i = 0; i < Width * Width; i++)
	{
		aM[i] = i+1;
		aN[i] = i+ Width * Width+ 1;
	}

	//출력
	std::cout << std::endl;
	matrixPrint(aM, Width);
	std::cout << " (*) " << std::endl;
	matrixPrint(aN, Width);
	std::cout << " = " << std::endl;


	GPUACC gpuacc;
	gpuacc.MatrixMultiplication(aM, aN, aP, Width);

	matrixPrint(aP, Width);

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