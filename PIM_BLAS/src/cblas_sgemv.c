#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sgemv(const enum CBLAS_ORDER Order,
	const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
	const float alpha, const float *A, const int lda,
	const float *X, const int incX, const float beta,
	float *Y, const int incY) {
	float tempA[256] = { 0 };
	float *pA = tempA;
	if (M < 16 || N < 16) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempA[16 * i + j] = A[N*i + j];
			}
		}
	}
	float(*array2)[16] = (float(*)[16]) tempA;
	float OrderChange[16][16], transpose[16][16] = { {0} };
	if (Order == 102) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				OrderChange[i][j] = array2[j][i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempA[16 * i + j] = OrderChange[i][j];
			}
		}
	}
	if (TransA == 112 || TransA == 113) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				transpose[i][j] = array2[j][i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempA[16 * i + j] = transpose[i][j];
			}
		}
	}
	if (TransA == 113 || TransA == 114) {
		printf("Not Support Complex\n");
	}
	float tempAX[16] = { 0 };
	float tempX[16] = { 0 };
	float tempY[16] = { 0 };
	for (int i = 0; i < lda; i++) {
		tempX[i] = X[i*incX];
	}
	for (int i = 0; i < lda; i++) {
		tempAX[i] = MAC_16(pA, tempX);
		pA = pA + 16;
	}
	SIMD_SCAL_MUL_16(alpha, tempAX);
	for (int i = 0; i < lda; i++) {
	tempY[i] = Y[i*incY];
	}
	SIMD_SCAL_MUL_16(beta, tempY);
	SIMD_ADD_16(tempAX, tempY);
	for (int i = 0; i < lda; i++) {
		Y[i*incY] = tempY[i];
	}
	printf("escal cblas_sgemv enabled!! \n");
}
