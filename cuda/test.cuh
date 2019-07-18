#pragma once

#pragma comment(lib, "cudart.lib")
#if _DEBUG
#pragma comment(lib, "opencv_world330d.lib")
#else
#pragma comment(lib, "opencv_world330.lib")
#endif

#ifdef __CUDACC__
#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#endif

#ifdef __INTELLISENSE__
int __float_as_int(float in);
float __int_as_float(int in);
short __float2half_rn(float in);

//Compare-and-Swap operation.
unsigned int atomicInc(unsigned int* address, unsigned int val);
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address, unsigned long long int compare, unsigned long long int val);

int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val);
float atomicAdd(float* address, float val); double atomicAdd(double* address, double val);

int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address, unsigned int val);


#define __syncthreads()
#define __syncthreads_or(a) a

#endif