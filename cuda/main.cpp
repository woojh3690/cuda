#include <iostream>
#include "GPUACC.cuh"

int main(int argc, char* argv[])
{
	std::cout << "hello world" << std::endl;
	float aM = 3;
	float aN = 4;
	float aP;
	int Width = 10;

	GPUACC gpuacc;
	gpuacc.MatrixMultiplication(&aM, &aN, &aP, Width);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "maxGridSize : " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;

	return 0;
}