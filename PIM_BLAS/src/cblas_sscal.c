#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sscal(const int N, const float alpha, float *X, const int incX) {
    int size_x = N * incX;
    printf("%d\n", size_x);
    /*if(incX==1) {
        for(int i=0; i<N; i=i+16) {
            SIMD_SCAL_MUL_16(alpha, X);
            X = X+16;
        } 
    }*/ //size 추가로 인한 주석 처리
    if(incX>=1) {
        float tempX[16];
        for(int i=0; i<N; i=i+16) {
            for(int j=0; j<16; j++) {
                if((i+j)*incX >= size_x) break;
                tempX[j] = X[(i+j)*incX];
            }
            SIMD_SCAL_MUL_16(alpha, tempX);
            for(int j=0; j<16; j++) {
                if((i+j)*incX >= size_x) break;
                X[(i+j)*incX] = tempX[j];
            }
        }
    } else {
        assert(0);
    }
    printf("escal cblas enabled!! \n");
}
