#include "GPUACC.cuh"
#define FHd_THREADS_PER_BLOCK 4
#define MU_THREADS_PER_BLOCK 4

int cos(float temp)
{
	return 1;
}

int sin(float temp)
{
	return 1;
}

void Q() {
	int M = 10;
	int N = 10;

	int phiMag[] = { 1,2,3 };
	int rPhi[] = { 1,2,3 };
	int iPhi[] = { 1,2,3 };

	int rQ[] = { 1,2,3 };
	int iQ[] = { 1,2,3 };

	int expQ[] = {1,2,3};


	//------------CPU 순차 A코드----------------
	for (int m = 0; m < M; m++)
	{
		phiMag[m] = rPhi[m] * rPhi[m] +
			iPhi[m] * iPhi[m];

		for (int n = 0; n < N; n++)
		{
			rQ[n] += phiMag[m] * cos(expQ);
			iQ[n] += phiMag[m] * sin(expQ);
		}
	}

	//------------CPU 순차 B코드----------------
	for (int m = 0; m < M; M++)
	{
		rMu[m] = rPhi[m] * rD[m] +
			iPhi[m] * iD[m];
		iMu[m] = rPhi[m] * iD[m] +
			iPhi[m] * rD[m];

		for (int n = 0; n < N; n++)
		{
			float expFHd = 2 * PI * (kx[m] * x[n]) + ky[m] * y[n] + kz[m] * z[n]);

			float cArg = cos(expFHd);
			float sArg = sin(expFHd);

			rFHd[n] += rMu[m] * cArg -
				iMu[m] * sArg;
			iFHd[n] += iMu[m] * cArg +
				rMu[m] * sArg;

		}
	}
}

//B코드 GPU kernel코드 2개로 변환
__global__ void cmpMu(float* rPhi, float* iPhi, float* rD, float* iD, float* rMu, float* iMu)
{
	int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;

	rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];
	iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
}

__global__ void cmpFHd(float* rPhi, float* iPhi, float* phiMag,
	float* kx, float* ky, float* kz, float* x, float* y, float* z, float* rMu, float* iMu, int M)
{
	int n = blockIdx.x * FHd_THREADS_PER_BLOCK + threadIdx.x;

	for (int m = 0; m < M; m++)
	{
		float expFHd = 2 * 3.14 * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);

		float cArg = cos(expFHd);
		float sArg = sin(expFHd);

		rFHd[n] += rMu[m] * cArg - iMu[m] * sArg;
		iFHd[n] += iMu[m] * cArg + rMu[m] * sArg;
	}

}