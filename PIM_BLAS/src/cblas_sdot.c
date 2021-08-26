#include "pim_avail_op.h"

float cblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY) 
{
    float sdot_result = 0;
    float tempX[16], tempY[16];
    int size_x = N * incX;
    int size_y = N * incY;

    //stride<0인 경우 추가?
    if(incX>0 && incY>0) {
        for(int i=0; i<N; i=i+16){
            for(int j=0; j<16; j++){
                tempX[j] = ((i+j)*incX >= size_x) ? 0 : X[(i+j)*incX];
                tempY[j] = ((i+j)*incY >= size_y) ? 0 : Y[(i+j)*incY];
            }
            sdot_result += MAC_16(tempX, tempY);
        }
    }
    else {
        assert(0);
    }
    return sdot_result;
}
