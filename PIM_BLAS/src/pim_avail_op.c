#include <stdio.h>
#include <assert.h>
#include "pim_avail_op.h"
#ifdef SIMD

int SIMD_CYCLE = 0;
int MV_CYCLE = 0;

void SIMD_ADD_16(float* x, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", x);
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector(%p, %p, 16, 0)\n", x, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", x);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, add)\n", x, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", x);
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_op(%p, %p, 16, 0)\n", x, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] + y[i];
    }
    SIMD_CYCLE++;
}

void SIMD_SUB_16(float* x, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", x);
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector(%p, %p, 16, 1)\n", x, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", x);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, sub)\n", x, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", x);
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_op(%p, %p, 16, 1)\n", x, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] - y[i];
    }
    SIMD_CYCLE++;
}

void SIMD_MUL_16(float* x, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", x);
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector(%p, %p, 16, 2)\n", x, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", x);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", x);
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_op(%p, %p, 16, 2)\n", x, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] * y[i];
    }
    SIMD_CYCLE+=2;
}

void SIMD_DIV_16(float* x, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", x);
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector(%p, %p, 16, 3)\n", x, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", x);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, div)\n", x, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", x);
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_op(%p, %p, 16, 3)\n", x, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] / y[i];
    }
    SIMD_CYCLE+=4;
}

void SIMD_SCAL_ADD_16(float a, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector_imm(%f, %p, 16, 0)\n", a, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		float imm = a;
		printf("dpim.set(%p, 16)\n", &a);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_imm(%f, %p, 16, 0)\n", a, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] + a;
    }
    SIMD_CYCLE++;
}

void SIMD_SCAL_SUB_16(float a, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector_imm(%f, %p, 16, 1)\n", a, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		float imm = a;
		printf("dpim.set(%p, 16)\n", &a);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, sub)\n", &a, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_imm(%f, %p, 16, 1)\n", a, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] - a;
    }
    SIMD_CYCLE++;
}

void SIMD_SCAL_MUL_16(float a, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector_imm(%f, %p, 16, 2)\n", a, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		float imm = a;
		printf("dpim.set(%p, 16)\n", &a);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, mul)\n", &a, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_imm(%f, %p, 16, 2)\n", a, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] * a;
    }
    SIMD_CYCLE+=2;
}

void SIMD_SCAL_DIV_16(float a, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector_imm(%f, %p, 16, 3)\n", a, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		float imm = a;
		printf("dpim.set(%p, 16)\n", &a);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, div)\n", &a, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", y);
		printf("pnm.vector_imm(%f, %p, 16, 3)\n", a, y);
		printf("pnm.store(%p, 16)\n", y);
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] / a;
    }
    SIMD_CYCLE+=4;
}

float MAC_16(float* v_a, float* v_b) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", v_a);
		printf("apim.load(%p, 16)\n", v_b);
		printf("apim.vector(%p, %p, 16, 2)\n", v_a, v_b);
		printf("apim.vector_acc(%p, 16)\n", v_b);
		printf("apim.store(%p, 1)\n",v_b);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", v_a);
		printf("dpim.set(%p, 16)\n", v_b);
		printf("dpim.vector_comp(%p, %p, 16, mul)\n", v_a, v_b);
		printf("dpim.vector_acc(%p, 16)\n", v_b);
		printf("dpim.store(%p, 1)\n",v_b);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", v_a);
		printf("pnm.load(%p, 16)\n", v_b);
		printf("pnm.vector_op(%p, %p, 16, 2)\n", v_a, v_b);
		printf("pnm.vector_acc(%p, 16)\n", v_b);
		printf("pnm.store(%p, 1)\n", v_b);
#endif
    float rt = 0;
    for(int i=0; i<16; i++) {
        rt += v_a[i] * v_b[i];
    }
    SIMD_CYCLE+=3;
    return rt;
}

float ACC_16(float* v_a) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", v_a);
		printf("apim.vector_acc(%p, 16)\n", v_a);
		printf("apim.store(%p, 1)\n", v_a);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", v_a);
		printf("dpim.vector_acc(%p, 16)\n", v_a);
		printf("dpim.store(%p, 1)\n", v_a);
#endif
#ifdef PNM
		printf("pnm.load(%p, 16)\n", v_a);
		printf("pnm.vector_acc(%p, 16)\n", v_a);
		printf("pnm.store(%p, 1)\n", v_a);
#endif
    float rt = 0;
    for(int i=0; i<16; i++) {
        rt += v_a[i];
    }
    SIMD_CYCLE+=2;
    return rt;
}

void USE_SHARED_MEM() {
    assert(0);
}

#endif 

#ifndef SIMD
// matrix&vector level operations (simple blas)
float VEC_ACC(float *x, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
			printf("apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
			printf("apim.vector_acc(%p, %d)\n", x+i, APIM_SIMD_SIZE);
		}
		temp/=APIM_SIMD_SIZE;
	}
	printf("apim.store(%p, 1)\n", x);
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
			printf("dpim.set(%p, %d)\n", x, DPIM_SIMD_SIZE);
			printf("dpim.vector_acc(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
		}
		temp/=DPIM_SIMD_SIZE;
	}
	printf("dpim.store(%p, 1)\n", x);
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
			printf("pnm.load(%p, %d)\n", x+i, PNM_SIMD_SIZE);
			printf("pnm.vector_acc(%p, %d)\n", x+i, PNM_SIMD_SIZE);
		}
		temp/=PNM_SIMD_SIZE;
	}
	printf("pnm.store(%p, 1)\n", x);
#endif
	float rt = 0;
	for(int i=0; i<width; i++) {
		rt += x[i];
	}
  MV_CYCLE++;
	return rt;
}

void VEC_ADD(float *x, float *y, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
			printf("apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
			printf("apim.load(%p, %d)\n", y+i, APIM_SIMD_SIZE);
			printf("apim.vector(%p, %p, %d, 0)\n", x+i, y+i, APIM_SIMD_SIZE);
			printf("apim.store(%p, %d)\n", y+i, APIM_SIMD_SIZE);
		}
		temp/=APIM_SIMD_SIZE;
	}
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
			printf("dpim.set(%p, %d)\n", x, DPIM_SIMD_SIZE);
			printf("dpim.set(%p, %d)\n", y+i, DPIM_SIMD_SIZE);
			printf("dpim.vector_comp(%p, %p, %d, add)\n", x+i, y+i, DPIM_SIMD_SIZE);
			printf("dpim.store(%p, %d)\n", y+i, DPIM_SIMD_SIZE);
		}
		temp/=DPIM_SIMD_SIZE;
	}
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
			printf("pnm.load(%p, %d)\n", x+i, PNM_SIMD_SIZE);
			printf("pnm.load(%p, %d)\n", y+i, PNM_SIMD_SIZE);
			printf("pnm.vector(%p, %p, %d, 0)\n", x+i, y+i, PNM_SIMD_SIZE);
			printf("pnm.store(%p, %d)\n", y+i, PNM_SIMD_SIZE);
		}
		temp/=PNM_SIMD_SIZE;
	}
#endif
	for(int i=0; i<width; i++) {
		y[i] += x[i];
	}
  MV_CYCLE++;
}
void VEC_MUL(float *x, float *y, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
			printf("apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
			printf("apim.load(%p, %d)\n", y+i, APIM_SIMD_SIZE);
			printf("apim.vector(%p, %p, %d, 2)\n", x+i, y+i, APIM_SIMD_SIZE);
			printf("apim.store(%p, %d)\n", y+i, APIM_SIMD_SIZE);
		}
		temp/=APIM_SIMD_SIZE;
	}
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
			printf("dpim.set(%p, %d)\n", x, DPIM_SIMD_SIZE);
			printf("dpim.set(%p, %d)\n", y+i, DPIM_SIMD_SIZE);
			printf("dpim.vector_comp(%p, %p, %d, mul)\n", x+i, y+i, DPIM_SIMD_SIZE);
			printf("dpim.store(%p, %d)\n", y+i, DPIM_SIMD_SIZE);
		}
		temp/=DPIM_SIMD_SIZE;
	}
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
			printf("pnm.load(%p, %d)\n", x+i, PNM_SIMD_SIZE);
			printf("pnm.load(%p, %d)\n", y+i, PNM_SIMD_SIZE);
			printf("pnm.vector(%p, %p, %d, 2)\n", x+i, y+i, PNM_SIMD_SIZE);
			printf("pnm.store(%p, %d)\n", y+i, PNM_SIMD_SIZE);
		}
		temp/=PNM_SIMD_SIZE;
	}
#endif
	for(int i=0; i<width; i++) {
		y[i] *= x[i];
	}
  MV_CYCLE+=2;
}
void VEC_IMM_ADD(float imm, float *x, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
			printf("apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
			printf("apim.vector_imm(%f, %p, %d, 0)\n", imm, x+i, APIM_SIMD_SIZE);
			printf("apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);
		}
		temp/=APIM_SIMD_SIZE;
	}
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
			printf("dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
			printf("dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
			printf("dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
			printf("dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
		}
		temp/=DPIM_SIMD_SIZE;
	}
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
			printf("pnm.load(%p, %d)\n", x+i, PNM_SIMD_SIZE);
			printf("pnm.vector_imm(%f, %p, %d, 0)\n", imm, x+i, PNM_SIMD_SIZE);
			printf("pnm.store(%p, %d)\n", x+i, PNM_SIMD_SIZE);
		}
		temp/=PNM_SIMD_SIZE;
	}
#endif
	for(int i=0; i<width; i++) {
		x[i] += imm;
	}
  MV_CYCLE++;
}
void VEC_IMM_MUL(float imm, float *x, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
			printf("apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
			printf("apim.vector_imm(%f, %p, %d, 2)\n", imm, x+i, APIM_SIMD_SIZE);
			printf("apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);
		}
		temp/=APIM_SIMD_SIZE;
	}
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
			printf("dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
			printf("dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
			printf("dpim.vector_comp(%p, %p, %d, mul)\n", &imm, x+i, DPIM_SIMD_SIZE);
			printf("dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
		}
		temp/=DPIM_SIMD_SIZE;
	}
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
			printf("pnm.load(%p, %d)\n", x+i, PNM_SIMD_SIZE);
			printf("pnm.vector_imm(%f, %p, %d, 2)\n", imm, x+i, PNM_SIMD_SIZE);
			printf("pnm.store(%p, %d)\n", x+i, PNM_SIMD_SIZE);
		}
		temp/=PNM_SIMD_SIZE;
	}
#endif
	for(int i=0; i<width; i++) {
		x[i] *= imm;
	}
  MV_CYCLE++;
}
void MV_MUL(float *x, float *A, float *y, int N, int M) {
#ifdef APIM
	int n = N/APIM_SIMD_SIZE;
	int m = M/APIM_SIMD_SIZE;
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			printf("apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
			printf("apim.xbar_init(%p)\n", A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
			printf("apim.mvmul(%p, Xbar, %d)\n", x+i*APIM_SIMD_SIZE, APIM_SIMD_SIZE);
			printf("apim.store(%p, %d)\n", y+i*APIM_SIMD_SIZE+j, APIM_SIMD_SIZE);
		}
	}
#endif
#ifdef DPIM
	int n = N/DPIM_SIMD_SIZE;
	int m = M/DPIM_GANG_SIZE;
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			printf("dpim.set(%p, %d)\n", x+i*DPIM_SIMD_SIZE, DPIM_SIMD_SIZE);
			printf("dpim.set(%p, %d)\n", A+i*DPIM_GANG_SIZE, DPIM_GANG_SIZE);
			printf("dpim.vector_comp(%p, %d, bank-parallel)\n", x+i*DPIM_SIMD_SIZE, DPIM_GANG_SIZE);
			printf("dpim.store(%p, %d)\n", y+i*DPIM_GANG_SIZE+j, DPIM_SIMD_SIZE);
		}
	}
#endif
#ifdef PNM
	int n = N/PNM_SIMD_SIZE;
	int m = M/PNM_SIMD_SIZE;
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){		
      printf("pnm.load(%p, %d)\n", x+i, PNM_SIMD_SIZE);
			printf("pnm.load(%p, %d)\n", A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE, PNM_SIMD_SIZE);
			printf("pnm.vector(%p, %p, %d, 2)\n", A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE, x+i*PNM_SIMD_SIZE, PNM_SIMD_SIZE);
			printf("pnm.store(%p, %d)\n", y+i*PNM_SIMD_SIZE+j, PNM_SIMD_SIZE);
		}
	}
#endif
	for(int i=0; i<N; i++){
		VEC_MUL(x, &A[i*M], N);
		y[i]=VEC_ACC(&A[i*M], N);
	}
  MV_CYCLE+=2;
}
void MAT_IMM_MUL(float imm, float *A, int N, int M) {
	for(int i=0; i<N; i++){
		VEC_IMM_MUL(imm, &A[i*M], M);
	}
}
void MM_ADD(float *A, float *B, int N, int M) {
	for(int i=0; i<N; i++){
		VEC_ADD(&A[i*M], &B[i*M], M);
	}
}
void MM_MUL(float* A, float *B, float *C, int N, int M, int K) {
	for(int i=0; i<N; i++){
		MV_MUL(&A[i*M], B, &C[i*K], M, K);
	}
}
#endif
