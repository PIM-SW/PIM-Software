#include "cblas.h"

int main()
{
    float y[32];
    for(int i=0; i<32; i++) {
        y[i] = i;
    }
    float alpha = 2.0;
   
    cblas_sscal(16, alpha, y, 2);
    for(int i=0; i<32; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");
    return 0;
}
