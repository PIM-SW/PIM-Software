#ifndef CBLAS_H
#define CBLAS_H
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */

typedef enum CBLAS_ORDER {
    CblasRowMajor=101,
    CblasColMajor=102
} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans=111,
    CblasTrans=112,
    CblasConjTrans=113,
    CblasConjNoTrans=114
} CBLAS_TRANSPOSE;


float cblas_sdsdot(const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY);

float cblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

void cblas_sscal(const int N, const float alpha, float *X, const int incX);

void cblas_sgemv(const enum CBLAS_ORDER Order,
                const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                const float alpha, const float *A, const int lda,
                const float *X, const int incX, const float beta,
                float *Y, const int incY);

void cblas_sgemv_usermode(const enum CBLAS_ORDER Order,
	                        const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
	                        const float alpha, const float *A, const int lda,
	                        const float *X, const int incX, const float beta,
	                        float *Y, const int incY, const int thread_group, const int block);

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const float alpha, const float *A,
                const int lda, const float *B, const int ldb,
                const float beta, float *C, const int ldc);
                
void cblas_sparselengthssum(float *A, const int *indices, const int *lengths, const int l);

float pimblas_sdsdot(const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY);

float pimblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);

void pimblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

void pimblas_sscal(const int N, const float alpha, float *X, const int incX);

void pimblas_sgemv(const enum CBLAS_ORDER Order,
                const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                const float alpha, const float *A, const int lda,
                const float *X, const int incX, const float beta,
                float *Y, const int incY);

void pimblas_sgemv_usermode(const enum CBLAS_ORDER Order,
	                        const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
	                        const float alpha, const float *A, const int lda,
	                        const float *X, const int incX, const float beta,
	                        float *Y, const int incY, const int thread_group, const int block);

void pimblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const float alpha, const float *A,
                const int lda, const float *B, const int ldb,
                const float beta, float *C, const int ldc);
                
void pimblas_sparselengthssum(float *A, const int *indices, const int *lengths, const int l);

void testhello();

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif
