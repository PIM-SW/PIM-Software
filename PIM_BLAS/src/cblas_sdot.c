#include "pim_avail_op.h"
#include "cblas.h"

float cblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY) 
{
#ifdef PRINT
	printf("%s\n",__FILE__);
#endif
    float sdot_result = 0;
    float tempX[16], tempY[16];
    int size_x = N * incX;
    int size_y = N * incY;

    if(incX>0 && incY>0) {
#ifdef SIMD
        for(int i=0; i<N; i=i+16){
            for(int j=0; j<16; j++){
                tempX[j] = ((i+j)*incX >= size_x) ? 0 : X[(i+j)*incX];
                tempY[j] = ((i+j)*incY >= size_y) ? 0 : Y[(i+j)*incY];
            }
            sdot_result += MAC_16(tempX, tempY);
        }
#endif
#ifndef SIMD
			if(incX==1 && incY==1) {
				VEC_MUL(X, Y, N);
				sdot_result = VEC_ACC(Y, N);
			}
#endif
    }
    else {
        assert(0);
    }
    return sdot_result;
}
