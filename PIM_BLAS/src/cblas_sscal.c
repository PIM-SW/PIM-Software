#include "pim_avail_op.h"

void cblas_sscal(const int N, const float alpha, float *X, const int incX) {
    if(incX==1) {
        for(int i=0; i<N; i=i+16) {
            SIMD_SCAL_MUL_16(alpha, X);
            X = X+16;
        }
    } else if(incX>1) {
        float tempX[16];
        for(int i=0; i<N; i=i+16) {
            for(int j=0; j<16; j++) {
                tempX[j] = X[(i+j)*incX];
            }
                SIMD_SCAL_MUL_16(alpha, tempX);
            for(int j=0; j<16; j++) {
                X[(i+j)*incX] = tempX[j];
            }
        }
    } else {
        assert(0);
    }
}
