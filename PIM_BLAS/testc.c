#include "cblas.h"
#include "pim_avail_op.h"

int main()
{
    LANG_NUM = 0;
    int a0=0, a1=0, a2=0, a3=0, a4=0, a5=0;
    float x[36];
    for(int j=0; j<36; j++) {
        x[j] = j%16;
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
    float X[16*3];
    for(int i=0; i<18; i++) {
        X[i] = 9*i;
    }
    float Y[16*2];
    for(int i=0; i<12; i++) {
        Y[i] = 4*i;
    }

    printf("\n\n/***********************************************************************************\n*\n");
    printf("* (c) Copyright 2021-2023 Yonsei University, Seoul, Korea.\n* All rights reserved.\n");
    printf("*\n* PIM-SW Framework is available in GitHub: https://github.com/PIM-SW\n*\n");
    printf("/***********************************************************************************\n\n\n");
    
//cblas_sscal
    //printf("\nresult of cblas_sscal\n");
    for(int i=0; i<1; i++) {
      cblas_sscal(36, alpha, y, 1);
      //a0++;
    }

//cblas_sdsdot
    printf("result of cblas_sdsdot\n");
    for(int i=0; i<1; i++) {
      cblas_sdsdot(18, alpha, x, 1, y, 1);
      a1++;
    }

//cblas_sdot
    printf("result of cblas_sdot\n");
    for(int i=0; i<1; i++) {
      cblas_sdot(18, x, 1, y, 1);
      a2++;
    }  

//cblas_saxpy
    printf("result of cblas_saxpy\n");
    for(int i=0; i<1; i++) {
      cblas_saxpy(18, alpha, x, 1, y, 1);
      a3++;
    }  

//cblas_sgemv
    printf("result of cblas_sgemv\n");
    for(int i=0; i<1; i++) {
      cblas_sgemv(CblasColMajor, CblasNoTrans, 6, 6, alpha, A, 6, X, 3, beta, Y, 2);
      a4++;
    }
 /*
//cblas_sgemv_usermode
    printf("\nresult of cblas_sgemv_usermode\n");
    for(int i=0; i<1; i++) {
      cblas_sgemv_usermode(CblasColMajor, CblasNoTrans, 16, 16, alpha, A, 16, X, 3, beta, Y, 2, 4, 1);
    }*/
    
//cblas_sgemm
    printf("result of cblas_sgemm\n");
    for(int i=0; i<1; i++) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 6, 6, 6, alpha, A, 6, B, 6, beta, C, 6);
      a5++;
    }
    /*
    printf("\ncblas_sscal count: %d\n", a0);
    printf("cblas_sdsdot count: %d\n", a1);
    printf("cblas_sdot count: %d\n", a2);
    printf("cblas_saxpy count: %d\n", a3);
    printf("cblas_sgemv count: %d\n", a4);
    printf("cblas_sgemm count: %d\n\n", a5);

    #ifdef SIMD
    printf("cblas_sscal cycle (mul*2): %d\n", a0*4);
    printf("cblas_sdsdot cycle (mac*2): %d\n", a1*6);
    printf("cblas_sdot cycle (mac*2): %d\n", a2*6);
    printf("cblas_saxpy cycle (mul*2+add*2): %d\n", a3*6);
    printf("cblas_sgemv cycle (mac*6+mul*2+add*1): %d\n", a4*23);
    printf("cblas_sgemm cycle (mac*36+mul*12+add*6): %d\n", a5*138);
    #endif

    #ifndef SIMD
    printf("cblas_sscal cycle (imm*1): %d\n", a0);
    printf("cblas_sdsdot cycle (mul*1+acc*1): %d\n", a1*3);
    printf("cblas_sdot cycle (mul*1+acc*1): %d\n", a2*3);
    printf("cblas_saxpy cycle (imm*1+add*1): %d\n", a3*2);
    printf("cblas_sgemv cycle (mul*7+acc*6+add*1+imm*1): %d\n", a4*22);
    printf("cblas_sgemm cycle (mul*42+acc*36+add*6+imm*12): %d\n", a5*138);
    #endif

//cblas_sparselengthssum
    int indices[8] = {0,2,4,6,8,10,12,14};
    int lengths[4] = {1,3,3,1};
    printf("\nresult of cblas_sparselenthssum\n");
    for(int i=0; i<1; i++) {
      cblas_sparselengthssum(A, indices, lengths, 4);
    }

#ifdef SIMD
    printf("\nSIMD_IR CYCLE: %d\n\n", SIMD_CYCLE);
#endif
#ifndef SIMD
    printf("\nBLAS_IR CYCLE: %d\n\n", MV_CYCLE);
#endif*/
    printf("#..Successfully generated IR trace: C_SIMD-to-IR_APIM_ISSAC.trace\n");
    printf("#..Successfully generated IR trace: C_SIMD-to-IR_APIM_PRIME.trace\n");
    printf("#..Successfully generated IR trace: C_SIMD-to-IR_DPIM_Newton.trace\n");
    printf("#..Successfully generated IR trace: C_SIMD-to-IR_DPIM_HBM-PIM.trace\n");
    printf("#..Successfully generated IR trace: C_SIMD-to-IR_PNM_RecNMP.trace\n");
    printf("#..Successfully generated IR trace: C_SIMD-to-IR_PNM_TensorDIMM.trace\n");
    printf("#..Successfully generated IR trace: C_Func-to-IR_APIM_ISSAC.trace\n");
    printf("#..Successfully generated IR trace: C_Func-to-IR_APIM_PRIME.trace\n");
    printf("#..Successfully generated IR trace: C_Func-to-IR_DPIM_Newton.trace\n");
    printf("#..Successfully generated IR trace: C_Func-to-IR_DPIM_HBM-PIM.trace\n");
    printf("#..Successfully generated IR trace: C_Func-to-IR_PNM_RecNMP.trace\n");
    printf("#..Successfully generated IR trace: C_Func-to-IR_PNM_TensorDIMM.trace\n\n\n");
    
    return 0;
}