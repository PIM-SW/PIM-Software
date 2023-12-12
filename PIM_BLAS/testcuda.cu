#include "cblas.h"
#include "stdio.h"
#include "stdint.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "unistd.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__global__ void mykernel() {
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  printf("I am a thread %d, SM %d, warp %d, lane %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}

int main()
{
    float x[36];
    int i,j;
    for(j=0; j<36; j++) {
        x[j] = 35-j;
    }

    float y[36];
    for(i=0; i<36; i++) {
        y[i] = i;
    }

    float alpha = 2.0;
    float beta = 7.0;

//cblas_sscal
    cblas_sscal(18, alpha, y, 2);
    printf("result of cblas_sscal\n");
    for(i=0; i<36; i++) {
        printf("%f \n", y[i]);
    }
    printf("\nShow me the CUDA thread number :\n");
    mykernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    printf("\n");

//cblas_sdsdot
    cblas_sdsdot(18, alpha, x, 2, y, 2);
    printf("result of cblas_sdsdot\n");
    for(i=0; i<36; i++) {
        printf("%f \n", y[i]);
    }
    printf("\nShow me the CUDA thread number :\n");
    mykernel<<<4, 4>>>();
    cudaDeviceSynchronize();
    printf("\n"); // sdsdot 삭제

//cblas_sdot
    cblas_sdot(18, x, 2, y, 2);
    printf("result of cblas_sdot\n");
    for(i=0; i<36; i++) {
        printf("%f \n", y[i]);
    }
    printf("\nShow me the CUDA thread number :\n");
    mykernel<<<4, 1>>>();
    cudaDeviceSynchronize();
    printf("\n");

//cblas_saxpy
    cblas_saxpy(18, alpha, x, 2, y, 2);
    printf("result of cblas_saxpy\n");
    for(i=0; i<36; i++) {
        printf("%f \n", y[i]);
    }
    printf("\nShow me the CUDA thread number :\n");
    mykernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("\n");

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

//cblas_sgemv
    cblas_sgemv(CblasColMajor, CblasNoTrans, 6, 6, alpha, A, 6, X, 3, beta, Y, 2);
    printf("result of cblas_sgemv\n");
    for(i=0; i<12; i++) {
        printf("%f \n", Y[i]);
    }
    printf("\nShow me the CUDA thread number :\n");
    mykernel<<<4, 4>>>();
    cudaDeviceSynchronize();
    printf("\n");
//cblas_sgemm
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 6, 6, 6, alpha, A, 6, B, 6, beta, C, 6);
    printf("result of cblas_sgemm\n");
    for(i=0; i<36; i++) {
        printf("%f \n", C[i]);
    }
    printf("\nShow me the CUDA thread number :\n");
    mykernel<<<4, 4>>>();
    cudaDeviceSynchronize();
    printf("\n");
    printf("testcuda Successfully Executed !!! \n");

    return 0;
}