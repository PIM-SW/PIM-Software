#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
	const enum CBLAS_TRANSPOSE TransB, int M, int N,
	int K, const float alpha, const float *A,
	const int lda, const float *B, const int ldb,
	const float beta, float *C, const int ldc) {
#ifdef PRINT
	printf("%s\n",__FILE__);
#endif
#ifdef SIMD
	float tempA[256] = { 0 }, tempB[256] = { 0 }, tempC[256] = { 0 };
	float *pA = tempA, *pB = tempB, *pC = tempC;
  if(M>16) {
    M = 16;
  }
  if(N>16) {
    N = 16;
  }
  if(K>16) {
    K = 16;
  }
	if (M < 16 || K < 16) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				tempA[16 * i + j] = A[K*i + j];
			}
		}
	}
	if (K < 16 || N < 16) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < N; j++) {
				tempB[16 * i + j] = B[N*i + j];
			}
		}
	}
	if (M < 16 || N < 16) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempC[16 * i + j] = C[N*i + j];
			}
		}
	}
	float(*array2A)[16] = (float(*)[16]) tempA;
	float(*array2B)[16] = (float(*)[16]) tempB;
	float OrderChangeA[16][16], OrderChangeB[16][16],
		transposeA[16][16], transposeB[16][16];
	if (Order == 102) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				OrderChangeA[i][j] = array2A[j][i];
				OrderChangeB[i][j] = array2B[j][i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempA[16 * i + j] = OrderChangeA[i][j];
				tempB[16 * i + j] = OrderChangeB[i][j];
			}
		}
	}
	if (TransA == 112 || TransA == 113) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < M; j++) {
				transposeA[i][j] = array2A[j][i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				tempA[16 * i + j] = transposeA[i][j];
			}
		}
	}
	if (TransB == 111 || TransB == 114) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				transposeB[i][j] = array2B[j][i];
			}
		}
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < N; j++) {
				tempB[16 * i + j] = transposeB[i][j];
			}
		}
	}
	if (TransA == 113 || TransA == 114) {
		printf("Not Support Complex\n");
	}
	if (TransB == 113 || TransB == 114) {
		printf("Not Support Complex\n");
	}
	float tempAB[256] = { 0 };
	float *pAB = tempAB;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < ldb; j++) {
			tempAB[16 * i + j] = MAC_16(pA, pB);
			pB = pB + 16;
		}
		pB = pB - 16 * ldb;
		pA = pA + 16;
	}
	if (Order == 102) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < M; j++) {
				transposeA[i][j] = tempAB[16 * j + i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				tempAB[16 * i + j] = transposeA[i][j];
			}
		}
	}
	for (int i = 0; i < M; i++) {
		SIMD_SCAL_MUL_16(alpha, pAB);
		pAB = pAB + 16;
	}
	pAB = pAB - 16 * M;
	for (int i = 0; i < M; i++) {
		SIMD_SCAL_MUL_16(beta, pC);
		pC = pC + 16;
	}
	pC = pC - 16 * M;
	for (int i = 0; i < M; i++) {
		SIMD_ADD_16(pAB, pC);
		pAB = pAB + 16;
		pC = pC + 16;
	}
	pC = pC - 16 * M;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			C[N*i + j] = tempC[16 * i + j];
		}
	}
#endif
#ifdef BLAS
	float temp[M*K];
	MM_MUL(A, B, temp, M, N, K);
	MAT_IMM_MUL(alpha, temp, M, K);
	MAT_IMM_MUL(beta, C, M, K);
	MM_ADD(temp, C, M, K);
#endif
}

void pimblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
	const enum CBLAS_TRANSPOSE TransB, int M, int N,
	int K, const float alpha, const float *A,
	const int lda, const float *B, const int ldb,
	const float beta, float *C, const int ldc) {
#ifdef PRINT
	printf("%s\n",__FILE__);
#endif
#ifdef SIMD
	float tempA[256] = { 0 }, tempB[256] = { 0 }, tempC[256] = { 0 };
	float *pA = tempA, *pB = tempB, *pC = tempC;
  if(M>16) {
    M = 16;
  }
  if(N>16) {
    N = 16;
  }
  if(K>16) {
    K = 16;
  }
	if (M < 16 || K < 16) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				tempA[16 * i + j] = A[K*i + j];
			}
		}
	}
	if (K < 16 || N < 16) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < N; j++) {
				tempB[16 * i + j] = B[N*i + j];
			}
		}
	}
	if (M < 16 || N < 16) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempC[16 * i + j] = C[N*i + j];
			}
		}
	}
	float(*array2A)[16] = (float(*)[16]) tempA;
	float(*array2B)[16] = (float(*)[16]) tempB;
	float OrderChangeA[16][16], OrderChangeB[16][16],
		transposeA[16][16], transposeB[16][16];
	if (Order == 102) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				OrderChangeA[i][j] = array2A[j][i];
				OrderChangeB[i][j] = array2B[j][i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tempA[16 * i + j] = OrderChangeA[i][j];
				tempB[16 * i + j] = OrderChangeB[i][j];
			}
		}
	}
	if (TransA == 112 || TransA == 113) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < M; j++) {
				transposeA[i][j] = array2A[j][i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				tempA[16 * i + j] = transposeA[i][j];
			}
		}
	}
	if (TransB == 111 || TransB == 114) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				transposeB[i][j] = array2B[j][i];
			}
		}
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < N; j++) {
				tempB[16 * i + j] = transposeB[i][j];
			}
		}
	}
	if (TransA == 113 || TransA == 114) {
		printf("Not Support Complex\n");
	}
	if (TransB == 113 || TransB == 114) {
		printf("Not Support Complex\n");
	}
	float tempAB[256] = { 0 };
	float *pAB = tempAB;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < ldb; j++) {
			tempAB[16 * i + j] = MAC_16(pA, pB);
			pB = pB + 16;
		}
		pB = pB - 16 * ldb;
		pA = pA + 16;
	}
	if (Order == 102) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < M; j++) {
				transposeA[i][j] = tempAB[16 * j + i];
			}
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				tempAB[16 * i + j] = transposeA[i][j];
			}
		}
	}
	for (int i = 0; i < M; i++) {
		SIMD_SCAL_MUL_16(alpha, pAB);
		pAB = pAB + 16;
	}
	pAB = pAB - 16 * M;
	for (int i = 0; i < M; i++) {
		SIMD_SCAL_MUL_16(beta, pC);
		pC = pC + 16;
	}
	pC = pC - 16 * M;
	for (int i = 0; i < M; i++) {
		SIMD_ADD_16(pAB, pC);
		pAB = pAB + 16;
		pC = pC + 16;
	}
	pC = pC - 16 * M;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			C[N*i + j] = tempC[16 * i + j];
		}
	}
#endif
#ifdef BLAS
	float temp[M*K];
	MM_MUL(A, B, temp, M, N, K);
	MAT_IMM_MUL(alpha, temp, M, K);
	MAT_IMM_MUL(beta, C, M, K);
	MM_ADD(temp, C, M, K);
#endif
}
