#include <stdio.h>
#include <assert.h>
#include "pim_avail_op.h"

void test() {
    printf("dd\n");
}

void SIMD_ADD_16(float* x, float* y) {
    printf("Add\n");
    for(int i=0; i<16; i++) {
        y[i] = x[i] + y[i];
    }
}

void SIMD_SUB_16(float* x, float* y) {
    printf("Sub\n");
    for(int i=0; i<16; i++) {
        y[i] = x[i] - y[i];
    }
}

void SIMD_MUL_16(float* x, float* y) {
    printf("Mul\n");
    for(int i=0; i<16; i++) {
        y[i] = x[i] * y[i];
    }
}

void SIMD_DIV_16(float* x, float* y) {
    printf("Div\n");
    for(int i=0; i<16; i++) {
        y[i] = x[i] / y[i];
    }
}

void SIMD_SCAL_ADD_16(float a, float* y) {
    printf("Scalar_Addl\n");
    for(int i=0; i<16; i++) {
        y[i] = y[i] + a;
    }
}

void SIMD_SCAL_SUB_16(float a, float* y) {
    printf("Scalar_Mul\n");
    for(int i=0; i<16; i++) {
        y[i] = y[i] - a;
    }
}

void SIMD_SCAL_MUL_16(float a, float* y) {
    printf("Scalar_Mul\n");
    for(int i=0; i<16; i++) {
        y[i] = y[i] * a;
    }
}

void SIMD_SCAL_DIV_16(float a, float* y) {
    printf("Scalar_Mul\n");
    for(int i=0; i<16; i++) {
        y[i] = y[i] / a;
    }
}

float MAC_16(float* v_a, float* v_b) {
    printf("Mul and Acc\n");
    float rt = 0;
    for(int i=0; i<16; i++) {
        rt += v_a[i] * v_b[i];
    }
    return rt;
}

float ACC_16(float* v_a) {
    printf("Acc\n");
    float rt = 0;
    for(int i=0; i<16; i++) {
        rt += v_a[i];
    }
    return rt;
}

void USE_SHARED_MEM() {
    printf("NOT YET\n");
    assert(0);
}

