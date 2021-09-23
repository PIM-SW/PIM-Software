#include "pim_avail_op.h"
#include "cblas.h"
double DMAC_16(double *X_d, double *Y_d);

float cblas_sdsdot(const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY)
{
    double sdsdot_result = 0;
    double alpha_double = (double) alpha;
    double tempX[16], tempY[16];

    //stride<0인 경우 추가?
    if(incX>0 && incY>0) {
        for(int i=0; i<N; i=i+16){
            for(int j=0; j<16; j++){
                tempX[j] = (double) X[(i+j)*incX];
                tempY[j] = (double) Y[(i+j)*incY];
            }
            sdsdot_result += DMAC_16(tempX, tempY);
        }
        sdsdot_result += alpha_double;
    }
    else {
        assert(0);
    }
    return (float) sdsdot_result;
}

double DMAC_16(double *X_d, double *Y_d){
    double rt = 0;
    for(int i=0; i<16; i++) {
        rt += X_d[i] * Y_d[i];
    }
    return rt;
}
