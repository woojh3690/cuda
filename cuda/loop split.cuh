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
//외부루프
__global__ void cmpMu(float* rPhi, float* iPhi, float* rD, float* iD, float* rMu, float* iMu)
{
	int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;

	rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];
	iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
}

//내부루프
//메모리 접근 대비 계산 비율 13:14
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

//레지스터를 적극 활용한 내부루프
//메모리 접근 대비 계산 비율 13:7
__global__ void cmpFHd(float* rPhi, float* iPhi, float* phiMag,
	float* kx, float* ky, float* kz, float* x, float* y, float* z, float* rMu, float* iMu, int M)
{
	int n = blockIdx.x * FHd_THREADS_PER_BLOCK + threadIdx.x;

	float xn_r = x[n]; float yn_r = y[n]; float zn_r = z[n];
	float rFHdn_r = rFHd[n]; float iFHdn_r = iFHd[n];

	//for 문 안에서 느린 전역메모리에 할당된 배열 x, y, z, rFHd, iFHd 에대한 접근이 많으므로
	//101, 102번째 줄에서 보이듯이 register에 할당되는 지역변수로 미리 할당한다.
	for (int m = 0; m < M; m++)
	{
		float expFHd = 2 * 3.14 * (kx[m] * xn_r + ky[m] * yn_r + kz[m] * zn_r);

		float cArg = cos(expFHd);
		float sArg = sin(expFHd);

		rFHdn_r += rMu[m] * cArg - iMu[m] * sArg;
		iFHdn_r += iMu[m] * cArg + rMu[m] * sArg;
	}

	rFHd[n] = rFHdn_r; iFHd[n] = iFHdn_r;
}