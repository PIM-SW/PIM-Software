#include <stdio.h>
#include <assert.h>
//#define SIMD

// #define PRINT
//#define APIM
//#define DPIM
//#define PNM

#define APIM_SIMD_SIZE 128
#define DPIM_SIMD_SIZE 16
#define DPIM_GANG_SIZE 512
#define PNM_SIMD_SIZE 128

int SIMD_CYCLE;
int MV_CYCLE;

#ifdef SIMD
void SIMD_ADD_16(float* x, float* y);
void SIMD_SUB_16(float* x, float* y);
void SIMD_MUL_16(float* x, float* y);
void SIMD_DIV_16(float* x, float* y);
void SIMD_SCAL_ADD_16(float a, float* y);
void SIMD_SCAL_SUB_16(float a, float* y);
void SIMD_SCAL_MUL_16(float a, float* y);
void SIMD_SCAL_DIV_16(float a, float* y);
float MAC_16(float* v_a, float* v_b);
float ACC_16(float* v_a);
void USE_SHARED_MEM();
#endif
#ifndef SIMD
// matrix&vector level operations (simple blas)
float VEC_ACC(float *x, int width);
void VEC_ADD(float *x, float *y, int width);
void VEC_MUL(float *x, float *y, int width);
void VEC_IMM_ADD(float imm, float *x, int width);
void VEC_IMM_MUL(float imm, float *x, int width);
void MV_MUL(float *x, float *A, float *y, int N, int M);
void MAT_IMM_MUL(float imm, float *A, int N, int M);
void MM_ADD(float *A, float *B, int N, int M);
void MM_MUL(float* A, float *B, float *C, int N, int M, int K);
#endif
