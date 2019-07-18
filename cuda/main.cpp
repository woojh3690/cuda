#include <iostream>
#include <ctime>
#include "GPUACC.cuh"
void matrixPrint(float* data, int width);

int main(int argc, char* argv[])
{
	std::cout << "hello world" << std::endl;

	int Width = 50;
	int size = Width * Width * sizeof(float);
	float* aM = (float*)malloc(size);
	float* aN = (float*)malloc(size);
	float* aP = (float*)malloc(size);

	//초기화
	srand((unsigned int)time(NULL));
	int max = 3;
	for (int i = 0; i < Width * Width; i++)
	{
		aM[i] = (rand() % max);
		aN[i] = (rand() % max);
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