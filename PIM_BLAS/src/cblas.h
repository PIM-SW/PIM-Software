#ifndef CBLAS_H
#define CBLAS_H
#include <stdio.h>

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

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const float alpha, const float *A,
                const int lda, const float *B, const int ldb,
                const float beta, float *C, const int ldc);
#endif
