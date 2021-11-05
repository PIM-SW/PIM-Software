#include "omp.h"
#include "cblas.h"
int main()
{
    float alpha = 0.1;
    float beta = -0.1;
    float A[20*100], B[100*20], C[20*20];
    float X[100*3], Y[20*2];
    int incX = 3;
    int incY = 2;
    for(int i=0; i<20*100; i++) {
        A[i] = i%19;
    } 
    for(int i=0; i<100*20; i++) {
        B[i] = i%17;
    }
    for(int i=0; i<20*20; i++) {
        C[i] = i;
    }
    for(int i=0; i<100*3; i++) {
        X[i] = i;
    }
    for(int i=0; i<20*2; i++) {
        Y[i] = i; 
    }
    //cblas_sgemv 20by100 matrix x 1
    cblas_sgemv(102, 112, 20, 100, alpha, A, 20, X, incX, beta, Y, incY);
    printf("\nresult of cblas_sgemv\n");
    for(int i=0; i<40; i++) {
        printf("%f\n", Y[i]);
    }
    printf("\n\n");
    //cblas_sgemv 1by100 matrices x 20
    float copyY[20] = { 0 };
    #pragma omp parallel num_threads(20)
    {
        #pragma omp for reduction (+:copyY)
        for(int h = 0; h < 20; h++) {
            for(int i=0; i<20*2; i++) {
                Y[i] = i;
            }
            cblas_sgemv(102, 112, 1, 100, alpha, A+100*h, 100, X, incX, beta, Y+2*h, incY);
            printf("thread %d result : %f\n", h, Y[2*h]);
            copyY[h] = Y[2*h];
        }
        printf("openmp parallel task loop num : %d\n", omp_get_thread_num());
    }
    for (int i = 0; i < 20; i++) {
        Y[incY*i] = copyY[i];
    }
    printf("\nresult of openmp cblas_sgemv\n");
    for(int i=0; i<40; i++) {
        printf("%f\n", Y[i]);
    }
    printf("\n\n");
    //cblas_sgemm 20by100 matrix x 1
    cblas_sgemm(101, 111, 111, 20, 20, 100, alpha, A, 100, B, 20, beta, C, 20);
    printf("\nresult of openmp cblas_sgemm");
    for(int i=0; i<20*20; i++) {
        if((i)%10==0) {
            printf("\n");
        }
        printf("%10.6f ", C[i]);
    }
    printf("\n\n");
    //cblas_sgemm 1by100 matrix x 20
    float copyC[20*20] = { 0 };
    #pragma omp parallel num_threads(20)
    {
        #pragma omp for reduction (+:C)
        for(int h = 0; h < 20; h++) {
            for(int i=0; i<20*20; i++) {
                C[i] = i;
            }
            cblas_sgemm(101, 111, 111, 1, 20, 100, alpha, A+100*h, 100, B, 20, beta, C+20*h, 20);
            for(int i = 0; i < 20; i++) {
                printf("thread %d result %d : %f\n", h, i, C[i+20*h]);
                copyC[i+20*h] = C[i+20*h];
            }
        }
        printf("openmp parallel task loop num : %d\n", omp_get_thread_num());
    }
    printf("\nresult of openmp cblas_sgemm");
    for(int i=0; i<20*20; i++) {
        if((i)%10==0) {
            printf("\n");
        }
        printf("%10.6f ", copyC[i]);
    }
    printf("\n\n");
   return 0;
}
