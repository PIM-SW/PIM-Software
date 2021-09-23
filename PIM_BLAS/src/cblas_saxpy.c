#include "pim_avail_op.h"
#include "cblas.h"

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY)
{
    float tempX[16], tempY[16];
    int size_y = N * incY;

    //stride<0인 경우 추가?
    if(incX>0 && incY>0) {
        for(int i=0; i<N; i=i+16){
            for(int j=0; j<16; j++){
                tempX[j] = X[(i+j)*incX];
                tempY[j] = Y[(i+j)*incY];
            }
            SIMD_SCAL_MUL_16(alpha, tempX);
            SIMD_ADD_16(tempX,tempY);
            for(int j=0; j<16; j++){
                if((i+j)*incY >= size_y) break;
                Y[(i+j)*incY] = tempY[j];
            }
        }
    }
    else {
        assert(0);
    }
}
