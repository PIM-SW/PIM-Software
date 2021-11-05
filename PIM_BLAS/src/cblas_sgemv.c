#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sgemv(const enum CBLAS_ORDER Order,
	const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
	const float alpha, const float *A, const int lda,
	const float *X, const int incX, const float beta,
	float *Y, const int incY) {
	for (int l = 0; l < ((M-1)/16) + 1; l++) {
		float copyY[16] = { 0 };
		for (int k = 0; k < ((N-1)/16) + 1; k++) {
			float tempA[256] = { 0 };
			float tempAX[16] = { 0 };
			float tempArow[16] = { 0 };
			float tempX[16] = { 0 };
			float tempY[16] = { 0 };
			int temp= 0; 
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					if( ((k*16 + j) < N) && ((l*16 + i) < M) ) {
							tempA[16 * i + j] = A[N*i + j + 16*k + 16*N*l];
					}
				}
			}
			float(*array2)[16] = (float(*)[16]) tempA;
			float OrderChange[16][16] = { {0} };
			float transpose[16][16] = { {0} };
			if (Order == 102) {
				for (int i = 0; i < 16; i++) {
					for (int j = 0; j < 16; j++) {
						OrderChange[i][j] = array2[j][i];
					}
				}
				for (int i = 0; i < 16; i++) {
					for (int j = 0; j < 16; j++) {
						tempA[16 * i + j] = OrderChange[i][j];
					}
				}
			}
			if (TransA == 112 || TransA == 113) {
				for (int i = 0; i < 16; i++) {
					for (int j = 0; j < 16; j++) {
						transpose[i][j] = array2[j][i];
					}
				}
				for (int i = 0; i < 16; i++) {
					for (int j = 0; j < 16; j++) {
						tempA[16 * i + j] = transpose[i][j];
					}
				}
			}
			if (TransA == 113 || TransA == 114) {
				printf("Not Support Complex\n");
			}
			if (Order == 102) {
				temp = k;
				k = l;
				l = temp;
			}
			if (TransA == 112 || TransA == 113) {
				temp = k;
				k = l;
				l = temp;
			}
			for (int i = 0; i < 16; i++) {
				if( (k*16 + i) < N ) {
					tempX[i] = X[(k*16 + i)*incX];
				}
			}
			if (Order == 102) {
				temp = k;
				k = l;
				l = temp;
			}
			if (TransA == 112 || TransA == 113) {
				temp = k;
				k = l;
				l = temp;
			}
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					tempArow[j] = tempA[16*i + j];
				}
				tempAX[i] = MAC_16(tempArow, tempX);
			}
			SIMD_SCAL_MUL_16(alpha, tempAX);
			for (int i = 0; i < 16; i++) {
				if( (l*16 + i) < M ) {
					tempY[i] = Y[(l*16 + i)*incY];
				}
			}
			if( k < 1) {
				SIMD_SCAL_MUL_16(beta, tempY);
			}
			else {
				SIMD_SCAL_MUL_16(0, tempY);
			}
			SIMD_ADD_16(tempAX, tempY);
			for (int i = 0; i < 16; i++) {
				if( l*16 + i < M ) {
					copyY[i] += tempY[i];
				}
			}
			printf("escal cblas_sgemv enabled!!\n");
		}
		for (int i = 0; i < 16; i++) {
			if( l*16 + i < M ) {
				Y[( l*16 + i) * incY ] = copyY[i];
			}
		}
	}
}