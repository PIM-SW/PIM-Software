#include "pim_avail_op.h"
#include "cblas.h"
void cblas_sparselengthssum(float *A, const int *indices, const int *lengths, const int l) {
#ifdef PRINT
	printf("%s\n",__FILE__);
#endif
        int temp = 0;  
        float tempA[16] = { 0, };
        float *pA = tempA; 
        float tempB[PNM_SIMD_SIZE] = { 0, };
        float *pB = tempB; 
#ifdef SIMD
        for(int k = 0; k<16; k++) {  
          for(int j = 0; j<l; j++) {             
            for(int i=temp; i<lengths[j]+temp; i++) {         
              tempA[i] = A[ indices[i] ];            
            }               
            temp += lengths[j];
            A[j] = ACC_16(pA);
            for(int i = 0; i<16; i++) {
              tempA[i] = 0;
            }                       
          }
          temp = 0;  
          for (int j = l; j<16; j++) {
            A[j] = 0;
          }     
          pA += 16;
          A += 16;
        }            
#endif
#ifdef BLAS   
        for(int k = 0; k<(256/PNM_SIMD_SIZE); k++) {  
          for(int j = 0; j<l; j++) {             
            for(int i=temp; i<lengths[j]+temp; i++) {         
              tempB[i] = A[ indices[i] ];            
            }               
            temp += lengths[j];
            A[j] = VEC_ACC(pB, PNM_SIMD_SIZE);         
            for(int i = 0; i<PNM_SIMD_SIZE; i++) {
              tempB[i] = 0;
            }                       
          }
          temp = 0;  
          for (int j = l; j<PNM_SIMD_SIZE; j++) {
            A[j] = 0;
          } 
          pB += PNM_SIMD_SIZE;    
          A += PNM_SIMD_SIZE;
        } 
#endif
}