#include "cblas.h"
#include "pim_avail_op.h"

int main()
{

    float x[36];
    for(int j=0; j<36; j++) {
        x[j] = 35-j;
    }

    float y[36];
    for(int i=0; i<36; i++) {
        y[i] = i;
    }

    float alpha = 2.0;
    float beta = 7.0;

    float A[36];
    for(int j=0; j<36; j++) {
        A[j] = 7*(35-j);
    }
    float B[36];
    for(int i=0; i<36; i++) {
        B[i] = 3*i;
    }
    float C[36];
    for(int i=0; i<36; i++) {
        C[i] = 3*i;
    }
    float X[18];
    for(int i=0; i<18; i++) {
        X[i] = 9*i;
    }
    float Y[12];
    for(int i=0; i<12; i++) {
        Y[i] = 4*i;
    }

//cblas_sscal
    printf("\nresult of cblas_sscal\n");
    for(int i=0; i<1000; i++) {
      cblas_sscal(18, alpha, y, 1);
    }

//cblas_sdsdot
    printf("result of cblas_sdsdot\n");
    for(int i=0; i<1000; i++) {
      cblas_sdsdot(18, alpha, x, 1, y, 1);
    }

//cblas_sdot
    printf("result of cblas_sdot\n");
    for(int i=0; i<1000; i++) {
      cblas_sdot(18, x, 1, y, 1);
    }  

//cblas_saxpy
    printf("result of cblas_saxpy\n");
    for(int i=0; i<1000; i++) {
      cblas_saxpy(18, alpha, x, 1, y, 1);
    }  

//cblas_sgemv
    printf("result of cblas_sgemv\n");
    for(int i=0; i<150; i++) {
      cblas_sgemv(CblasColMajor, CblasNoTrans, 6, 6, alpha, A, 6, X, 3, beta, Y, 2);
    }
    
//cblas_sgemm
    printf("result of cblas_sgemm\n");
    for(int i=0; i<150; i++) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 6, 6, 6, alpha, A, 6, B, 6, beta, C, 6);
    }

#ifdef SIMD
    printf("\nSIMD_IR CYCLE: %d\n\n", SIMD_CYCLE);
#endif
#ifndef SIMD
    printf("\nBLAS_IR CYCLE: %d\n\n", MV_CYCLE);
#endif

printf("testc Successfully Executed !!! \n");

    return 0;
}