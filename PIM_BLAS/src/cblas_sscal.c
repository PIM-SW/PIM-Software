#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sscal(const int N, const float alpha, float *X, const int incX) {
#ifdef PRINT
	printf("%s\n",__FILE__);
#endif
    int size_x = N * incX;
    if(incX>=1) {
#ifdef SIMD
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
#endif
#ifndef SIMD
			if(incX==1) {
				VEC_IMM_MUL(alpha, X, N);
			}
#endif
    } else {
        assert(0);
    }
}
