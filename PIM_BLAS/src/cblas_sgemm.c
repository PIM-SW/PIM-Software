#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
	const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
	const int K, const float alpha, const float *A,
	const int lda, const float *B, const int ldb,
	const float beta, float *C, const int ldc) {
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			float tempAB = 0;
			float tempC = 0;
			for (int k = 0; k < ((K-1)/16) + 1; k++) {
				float tempA[16] = { 0 };
				float tempB[16] = { 0 };
				for (int i = 0; i < 16; i++) {
					if( 16*k + i < K ) {
						if(Order == 101 && TransA == 111 && TransB == 111) {
								tempA[i] = A[lda*m + 16*k + i];
								tempB[i] = B[ldb*(16*k + i) + n];
						}
						else if(Order == 101 && TransA == 111 && TransB == 112) {
							tempA[i] = A[lda*m + 16*k + i];
							tempB[i] = B[ldb*n + 16*k + i];
						}
						else if(Order == 101 && TransA == 112 && TransB == 111) {
							tempA[i] = A[lda*(16*k + i) + n];
							tempB[i] = B[ldb*(16*k + i) + n];
						}
						else if(Order == 101 && TransA == 112 && TransB == 112) {
							tempA[i] = A[lda*(16*k + i) + n];
							tempB[i] = B[lda*n + 16*k + i];
						}
						else if (Order == 102 && TransA == 111 && TransB == 111) {
							tempA[i] = A[lda*(16*k + i) + n];
							tempB[i] = B[ldb*n + 16*k + i];
						}
						else if (Order == 102 && TransA == 111 && TransB == 112) {
							tempA[i] = A[lda*(16*k + i) + n];
							tempB[i] = B[ldb*(16*k + i) + n];
						}
						else if (Order == 102 && TransA == 112 && TransB == 111) {
							tempA[i] = A[lda*m + 16*k + i];
							tempB[i] = B[ldb*n + 16*k + i];
						}
						else if (Order == 102 && TransA == 112 && TransB == 112) {
							tempA[i] = A[lda*m + 16*k + i];
							tempB[i] = B[ldb*(16*k + i) + n];
						}
					}
					if (TransA == 113 || TransA == 114) {
						printf("Not Support Complex\n");
						assert(0);
					}
					if (TransB == 113 || TransB == 114) {
						printf("Not Support Complex\n");
						assert(0);
					}
				}
				tempAB += MAC_16(tempA, tempB);
			}
			if(Order == 101) {
				tempC = C[ldc*m + n];
				C[ldc*m + n] = alpha*tempAB + beta*tempC;
			}
			else if(Order == 102) {
				tempC = C[m + lda*n];
				C[m + lda*n] = alpha*tempAB + beta*tempC;
			}
		}
	}
	printf("\nescal cblas_sgemm enabled!!\n");
}