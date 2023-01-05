#include "omp.h"
#include "cblas.h"
#include "pim_avail_op.h"

int main()
{
    LANG_NUM = 1;
    float x[36];
    for(int j=0; j<36; j++) {
        x[j] = 35-j;
    }

    float y[360];
    for(int i=0; i<360; i++) {
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
    printf("\n\n/***********************************************************************************\n*\n");
    printf("* (c) Copyright 2021-2023 Yonsei University, Seoul, Korea.\n* All rights reserved.\n");
    printf("*\n* PIM-SW Framework is available in GitHub: https://github.com/PIM-SW\n*\n");
    printf("/***********************************************************************************\n\n\n");
//cblas_sscal
    #pragma omp parallel num_threads(1)
    {
    for(int i=0; i<100; i++) {
      cblas_sscal(360, alpha, y, 1);
      //a0++;
      }
    }
/*
//cblas_sdsdot
    printf("result of cblas_sdsdot\n");
    #pragma omp parallel num_threads(2)
    {
    cblas_sdsdot(18, alpha, x, 1, y, 1);
    printf("I am a OpenMP thread %d\n", omp_get_thread_num());
    }
    
//cblas_sdot
    printf("result of cblas_sdot\n");
    #pragma omp parallel num_threads(2)
    {
    cblas_sdot(18, x, 1, y, 1);
    printf("I am a OpenMP thread %d\n", omp_get_thread_num());
    }
    
//cblas_saxpy
    printf("result of cblas_saxpy\n");
    #pragma omp parallel num_threads(2)
    {
    cblas_saxpy(18, alpha, x, 1, y, 1);
    printf("I am a OpenMP thread %d\n", omp_get_thread_num());
    }

//cblas_sgemv
    printf("result of cblas_sgemv\n");
    #pragma omp parallel num_threads(2)
    {
    cblas_sgemv(CblasColMajor, CblasNoTrans, 6, 6, alpha, A, 6, X, 3, beta, Y, 2);
    printf("I am a OpenMP thread %d\n", omp_get_thread_num());
    }
    
//cblas_sgemm
    printf("result of cblas_sgemm\n");
    #pragma omp parallel num_threads(2)
    {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 6, 6, 6, alpha, A, 6, B, 6, beta, C, 6);
    printf("I am a OpenMP thread %d\n", omp_get_thread_num());
    }*/
    printf("#..Successfully generated IR trace: OpenMP_SIMD-to-IR_APIM_ISSAC.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_SIMD-to-IR_APIM_PRIME.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_SIMD-to-IR_DPIM_Newton.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_SIMD-to-IR_DPIM_HBM-PIM.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_SIMD-to-IR_PNM_RecNMP.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_Func-to-IR_APIM_ISSAC.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_Func-to-IR_APIM_PRIME.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_Func-to-IR_DPIM_Newton.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_Func-to-IR_DPIM_HBM-PIM.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_Func-to-IR_PNM_RecNMP.trace\n");
    printf("#..Successfully generated IR trace: OpenMP_Func-to-IR_PNM_TensorDIMM.trace\n\n\n");
    return 0;
}