#include <stdio.h>
#include <assert.h>

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

