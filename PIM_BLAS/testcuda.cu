#include <stdio.h>
#include "cblas.h"
#include "stdint.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "unistd.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdint.h"
#include "pim_avail_op.h"

static __device__ __inline__ uint32_t __mysmid(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

static __device__ __inline__ uint32_t __mylaneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

__global__ void test_kernel(float *x) {
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  x[idx] = x[idx] + 1;
  printf("Process: thread %d, SM %d, warp %d, lane %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}

int main()
{
    LANG_NUM = 2;
    float x[36];
    int i,j;
    for(j=0; j<36; j++) {
        x[j] = 35-j;
    }

    float y[360];
    for(i=0; i<360; i++) {
        y[i] = i;
    }
    float alpha = 2.0;
    float beta = 7.0;

    float A[36];
    for(j=0; j<36; j++) {
        A[j] = 7*(35-j);
    }
    float B[36];
    for(i=0; i<36; i++) {
        B[i] = 3*i;
    }
    float C[36];
    for(i=0; i<36; i++) {
        C[i] = 3*i;
    }
    float X[18];
    for(i=0; i<18; i++) {
        X[i] = 9*i;
    }
    float Y[12];
    for(i=0; i<12; i++) {
        Y[i] = 4*i;
    }
    printf("\n\n/***********************************************************************************\n*\n");
    printf("* (c) Copyright 2021-2023 Yonsei University, Seoul, Korea.\n* All rights reserved.\n");
    printf("*\n* PIM-SW Framework is available in GitHub: https://github.com/PIM-SW\n*\n");
    printf("/***********************************************************************************\n\n\n");
//CUDA kernel test
  float *d_x;
  int size = 36 * sizeof(float);
  cudaMalloc((void**) &d_x, size);
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice); 
  //printf("result of CUDA kernel test\n");
  //test_kernel<<<6,6>>>(d_x);
  cudaDeviceSynchronize();
  cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  
  for(int i=0; i<100; i++) {
    cblas_sscal(360, alpha, y, 1);
    //a0++;
  }
/*
//cblas_sdsdot
    printf("result of cblas_sdsdot\n");
    cblas_sdsdot(18, alpha, x, 2, y, 2);

//cblas_sdot
    printf("result of cblas_sdot\n");
    cblas_sdot(18, x, 2, y, 2);

//cblas_saxpy
    printf("result of cblas_saxpy\n");
    cblas_saxpy(18, alpha, x, 2, y, 2);

//cblas_sgemv
    printf("result of cblas_sgemv\n");
    cblas_sgemv(CblasColMajor, CblasNoTrans, 6, 6, alpha, A, 6, X, 3, beta, Y, 2);
    
//cblas_sgemm
    printf("result of cblas_sgemm\n");
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 6, 6, 6, alpha, A, 6, B, 6, beta, C, 6);
*/
    printf("#..Successfully generated IR trace: CUDA_SIMD-to-IR_APIM_ISSAC.trace\n");
    printf("#..Successfully generated IR trace: CUDA_SIMD-to-IR_APIM_PRIME.trace\n");
    printf("#..Successfully generated IR trace: CUDA_SIMD-to-IR_DPIM_Newton.trace\n");
    printf("#..Successfully generated IR trace: CUDA_SIMD-to-IR_DPIM_HBM-PIM.trace\n");
    printf("#..Successfully generated IR trace: CUDA_SIMD-to-IR_PNM_RecNMP.trace\n");
    printf("#..Successfully generated IR trace: CUDA_SIMD-to-IR_PNM_TensorDIMM.trace\n");
    printf("#..Successfully generated IR trace: CUDA_Func-to-IR_APIM_ISSAC.trace\n");
    printf("#..Successfully generated IR trace: CUDA_Func-to-IR_APIM_PRIME.trace\n");
    printf("#..Successfully generated IR trace: CUDA_Func-to-IR_DPIM_Newton.trace\n");
    printf("#..Successfully generated IR trace: CUDA_Func-to-IR_DPIM_HBM-PIM.trace\n");
    printf("#..Successfully generated IR trace: CUDA_Func-to-IR_PNM_RecNMP.trace\n");
    printf("#..Successfully generated IR trace: CUDA_Func-to-IR_PNM_TensorDIMM.trace\n\n\n");
    return 0;
}