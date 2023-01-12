#include "pim_avail_op.h"
#include "cblas.h"

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY)
{
#ifdef PRINT
	printf("%s\n",__FILE__);
#endif
    float tempX[16], tempY[16];
    int size_y = N * incY;

		if(incX>0 && incY>0) {
#ifdef SIMD
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
#endif
#ifdef BLAS
			if(incX==1 && incY==1) {
				VEC_IMM_MUL(alpha, X, N);
				VEC_ADD(X, Y, N);
			}
#endif
		}
		else {
			assert(0);
		}
}

void testhello() {
  printf("Hello World!\n");
}