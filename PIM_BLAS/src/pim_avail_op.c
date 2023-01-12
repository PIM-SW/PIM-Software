#include <stdio.h>
#include <assert.h>
#include "pim_avail_op.h"

int LANG_NUM;

int SIMD_CYCLE = 0;
int MV_CYCLE = 0;

int bg_pos, ba_pos, co_pos, ch_pos, ra_pos, ro_pos;
int bg_mask, ba_mask,  co_mask, ch_mask, ra_mask, ro_mask;
int ch, ra, bg, ba, co, ro;

int ti_pos, rb_pos, xb_pos;
int ti_mask, rb_mask, xb_mask; 
int ti, rb, xb;

int reg_pos, reg_mask, reg;

void ADDR(float* x) {
  int addr = (int) x;
  
#ifdef APIM
  #ifdef MAP_TYPE1 // Row-wise (ISSAC) tile - mac_unit - xbar
    ti_pos = 6, rb_pos = 10, xb_pos = 13;
    ti_mask = 15, rb_mask = 7,  xb_mask = 3;
  #endif
  #ifdef MAP_TYPE2 // Row-wise (PRIME) chip - reram_bank - xbar
    ti_pos = 6, rb_pos = 9, xb_pos = 12;
    ti_mask = 7, rb_mask = 7,  xb_mask = 3;
  #endif 
#endif

#ifdef PNM
  #ifdef MAP_TYPE1 // Row-wise (RecNMP)
    bg_pos = 6, ba_pos = 8, co_pos = 10, ch_pos = 20, ra_pos = 21, ro_pos = 22;
    bg_mask = 3, ba_mask = 3,  co_mask = 1023, ch_mask = 1, ra_mask = 1, ro_mask = 31;
  #endif
  #ifdef MAP_TYPE2 // Column-wise (TensorDIMM)
    ch_pos = 6, ra_pos = 7, bg_pos = 8, ba_pos = 10, co_pos = 12, ro_pos = 22;
    ch_mask = 1, ra_mask = 1, bg_mask = 3, ba_mask = 3, co_mask = 1023, ro_mask = 31;
  #endif 
#endif

#ifdef DPIM
  #ifdef MAP_TYPE1 // Row-wise (HBM-PIM)
    bg_pos = 6, ba_pos = 8, co_pos = 10, ch_pos = 13, ra_pos = 14, ro_pos = 15;
    bg_mask = 3, ba_mask = 3,  co_mask = 7, ch_mask = 1, ra_mask = 1, ro_mask = 31;
  #endif 
  #ifdef MAP_TYPE2 // Row-wise (Newton)
    bg_pos = 6, ba_pos = 8, co_pos = 10, ch_pos = 13, ra_pos = 14, ro_pos = 15;
    bg_mask = 3, ba_mask = 3,  co_mask = 7, ch_mask = 1, ra_mask = 1, ro_mask = 31;
  #endif
#endif

#ifdef USER_MODE
  if (THREAD_GROUP_SIZE >= 1) {
    bg_mask = 0, ba_mask = 0;
    rb_mask = 0, xb_mask = 0; 
  }
  if (THREAD_GROUP_SIZE == 2) {
    ba_mask = 1;
    rb_mask = 1;
  }
  if (THREAD_GROUP_SIZE >= 4) {
    ba_mask = 3;
    rb_mask = 3;
  }
  if (THREAD_GROUP_SIZE >= 8) {
    rb_mask = 7;
  }
  if (THREAD_GROUP_SIZE == 16) {
    ba_mask = 3;
    xb_mask = 1;
  }
  if (THREAD_GROUP_SIZE == 32) {
    ba_mask = 3;
    xb_mask = 3;
  }
  if (BLOCK_SIZE >= 1) {
    ch_mask = 0, ra_mask = 0;
    ti_mask = 0; bg_mask = 0;
  }
  if (BLOCK_SIZE >= 2) {
    ch_mask = 1; bg_mask = 1;
    ti_mask = 1;
  }
  if (BLOCK_SIZE == 4) {
    ra_mask = 1; bg_mask = 3;
    ti_mask = 3;
  }
  if (BLOCK_SIZE == 8) {
    ti_mask = 7;
  }
  if (BLOCK_SIZE == 16) {
    ti_mask = 15;
  }
#endif

  reg_pos = 3;
  reg_mask = 7;
  
  ch = (addr >> ch_pos) & ch_mask;
  ra = (addr >> ra_pos) & ra_mask;
  bg = (addr >> bg_pos) & bg_mask;
  ba = (addr >> ba_pos) & ba_mask;
  co = (addr >> co_pos) & co_mask;
  ro = (addr >> ro_pos) & ro_mask;
  
  ti = (addr >> ti_pos) & ti_mask;
  rb = (addr >> rb_pos) & rb_mask;
  xb = (addr >> xb_pos) & xb_mask;
  
  reg = (addr >> reg_pos) & reg_mask;
}

#ifdef SIMD
void SIMD_ADD_16(float* x, float* y) {
#ifdef APIM
    #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(x); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector(reg: %d, ", reg); 
    	ADDR(y); 
    	fprintf(fp0, " y: tile %d, mac_unit %d, x-bar %d, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp0, "apim.store(y: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(x); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector(reg: %d, ", reg); 
    	ADDR(y); 
    	fprintf(fp0, " y: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp0, "apim.store(y: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(x); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector(reg: %d, ", reg); 
    	ADDR(y); 
    	fprintf(fp0, " y: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp0, "apim.store(y: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb); 
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(x);
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector(reg: %d, ", reg); 
    	ADDR(y);
    	fprintf(fp1, " y: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp1, "apim.store(y: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
      fclose(fp1);
     } 
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(x);
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector(reg: %d, ", reg); 
    	ADDR(y);
    	fprintf(fp1, " y: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp1, "apim.store(y: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(x);
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector(reg: %d, ", reg); 
    	ADDR(y);
    	fprintf(fp1, " y: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp1, "apim.store(y: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
      fclose(fp1);
     }
    #endif  
#endif
#ifdef DPIM
  #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp2 = fopen("./IR/C_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(x);
      fprintf(fp2, "dpim.set(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp2, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 1) {
      FILE* fp2 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(x);
      fprintf(fp2, "dpim.set(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp2, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 2) {
      FILE* fp2 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(x);
      fprintf(fp2, "dpim.set(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp2, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
  #endif
  #ifdef MAP_TYPE2   
    if(LANG_NUM == 0) {
      FILE* fp3 = fopen("./IR/C_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(x);
      fprintf(fp3, "dpim.set(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp3, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3);
   }
   if(LANG_NUM == 1) {
      FILE* fp3 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(x);
      fprintf(fp3, "dpim.set(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp3, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3);
   }
   if(LANG_NUM == 2) {
      FILE* fp3 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(x);
      fprintf(fp3, "dpim.set(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp3, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3);
   }
  #endif     
#endif
#ifdef PNM
  #ifdef MAP_TYPE1
  if(LANG_NUM == 0) {
    FILE* fp4 = fopen("./IR/C_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(x);     
	  fprintf(fp4, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
     ADDR(y);
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  if(LANG_NUM == 1) {
    FILE* fp4 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(x);     
	  fprintf(fp4, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
     ADDR(y);
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
   if(LANG_NUM == 2) {
    FILE* fp4 = fopen("./IR/CUDA_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(x);     
	  fprintf(fp4, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
     ADDR(y);
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  #endif
  #ifdef MAP_TYPE2  
  if(LANG_NUM == 0) {
    FILE* fp5 = fopen("./IR/C_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(x);     
	  fprintf(fp5, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
     ADDR(y);
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 1) {
    FILE* fp5 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(x);     
	  fprintf(fp5, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
     ADDR(y);
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 2) {
    FILE* fp5 = fopen("./IR/CUDA_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(x);     
	  fprintf(fp5, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
     ADDR(y);
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
  #endif
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] + y[i];
    }
    SIMD_CYCLE++;
    //printf("simd add 16\n");
}
/*
void SIMD_SUB_16(float* x, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", x);
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector(%p, %p, 16, 1)\n", x, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", x);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, sub)\n", x, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
    ADDR(x);   
	  printf("pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y); 
    printf(" y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, 1)\n", ch, ra, bg, ba, ro, co); 
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] - y[i];
    }
    SIMD_CYCLE++;
    //printf("simd sub 16\n");
}*/

void SIMD_MUL_16(float* x, float* y) {
#ifdef APIM
   #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "apim.load(%p, 16)\n", x);
    	fprintf(fp0, "apim.load(%p, 16)\n", y);
	    fprintf(fp0, "apim.vector(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp0, "apim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "apim.load(%p, 16)\n", x);
    	fprintf(fp0, "apim.load(%p, 16)\n", y);
	    fprintf(fp0, "apim.vector(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp0, "apim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "apim.load(%p, 16)\n", x);
    	fprintf(fp0, "apim.load(%p, 16)\n", y);
	    fprintf(fp0, "apim.vector(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp0, "apim.store(%p, 16)\n", y);
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "apim.load(%p, 16)\n", x);
    	fprintf(fp1, "apim.load(%p, 16)\n", y);
	    fprintf(fp1, "apim.vector(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp1, "apim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "apim.load(%p, 16)\n", x);
    	fprintf(fp1, "apim.load(%p, 16)\n", y);
	    fprintf(fp1, "apim.vector(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp1, "apim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "apim.load(%p, 16)\n", x);
    	fprintf(fp1, "apim.load(%p, 16)\n", y);
	    fprintf(fp1, "apim.vector(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp1, "apim.store(%p, 16)\n", y);
      fclose(fp1);
     }
    #endif    
#endif
#ifdef DPIM   
      #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "dpim.set(%p, 16)\n", x);
    	fprintf(fp0, "dpim.set(%p, 16)\n", y);
	    fprintf(fp0, "dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp0, "dpim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "dpim.set(%p, 16)\n", x);
    	fprintf(fp0, "dpim.set(%p, 16)\n", y);
	    fprintf(fp0, "dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp0, "dpim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "dpim.set(%p, 16)\n", x);
    	fprintf(fp0, "dpim.set(%p, 16)\n", y);
	    fprintf(fp0, "dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp0, "dpim.store(%p, 16)\n", y);
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "dpim.set(%p, 16)\n", x);
    	fprintf(fp1, "dpim.set(%p, 16)\n", y);
	    fprintf(fp1, "dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp1, "dpim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "dpim.set(%p, 16)\n", x);
    	fprintf(fp1, "dpim.set(%p, 16)\n", y);
	    fprintf(fp1, "dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp1, "dpim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "dpim.set(%p, 16)\n", x);
    	fprintf(fp1, "dpim.set(%p, 16)\n", y);
	    fprintf(fp1, "dpim.vector_comp(%p, %p, 16, mul)\n", x, y);
		  fprintf(fp1, "dpim.store(%p, 16)\n", y);
      fclose(fp1);
     }
    #endif    
#endif
#ifdef PNM  
  #ifdef MAP_TYPE1
  if(LANG_NUM == 0) {
    FILE* fp4 = fopen("./IR/C_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(x);     
	  fprintf(fp4, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  if(LANG_NUM == 1) {
    FILE* fp4 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(x);     
	  fprintf(fp4, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
   if(LANG_NUM == 2) {
    FILE* fp4 = fopen("./IR/CUDA_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(x);     
	  fprintf(fp4, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  #endif
  #ifdef MAP_TYPE2  
  if(LANG_NUM == 0) {
    FILE* fp5 = fopen("./IR/C_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(x);     
	  fprintf(fp5, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 1) {
    FILE* fp5 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(x);     
	  fprintf(fp5, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 2) {
    FILE* fp5 = fopen("./IR/CUDA_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(x);     
	  fprintf(fp5, "pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
  #endif   
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] * y[i];
    }
    SIMD_CYCLE+=2;
    //printf("simd mul 16\n");
}
/*
void SIMD_DIV_16(float* x, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", x);
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector(%p, %p, 16, 3)\n", x, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		printf("dpim.set(%p, 16)\n", x);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, div)\n", x, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
    ADDR(x);   
	  printf("pnm.vector_op(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(y); 
    printf(" y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, 3)\n", ch, ra, bg, ba, ro, co); 
#endif
    for(int i=0; i<16; i++) {
        y[i] = x[i] / y[i];
    }
    SIMD_CYCLE+=4;
    //printf("simd div 16\n");
}*/

void SIMD_SCAL_ADD_16(float a, float* y) {
#ifdef APIM
   #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "apim.load(%p, 16)\n", y);
    	fprintf(fp0, "apim.vector_imm(%f, %p, 16, imm)\n", a, y);
    	fprintf(fp0, "apim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "apim.load(%p, 16)\n", y);
    	fprintf(fp0, "apim.vector_imm(%f, %p, 16, imm)\n", a, y);
    	fprintf(fp0, "apim.store(%p, 16)\n", y); 
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "apim.load(%p, 16)\n", y);
    	fprintf(fp0, "apim.vector_imm(%f, %p, 16, imm)\n", a, y);
    	fprintf(fp0, "apim.store(%p, 16)\n", y);
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "apim.load(%p, 16)\n", y);
    	fprintf(fp1, "apim.vector_imm(%f, %p, 16, imm)\n", a, y);
    	fprintf(fp1, "apim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "apim.load(%p, 16)\n", y);
    	fprintf(fp1, "apim.vector_imm(%f, %p, 16, imm)\n", a, y);
    	fprintf(fp1, "apim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "apim.load(%p, 16)\n", y);
    	fprintf(fp1, "apim.vector_imm(%f, %p, 16, imm)\n", a, y);
    	fprintf(fp1, "apim.store(%p, 16)\n", y);
      fclose(fp1);
     }
    #endif    
#endif
#ifdef DPIM
		float imm = a;   
   #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "dpim.set(%p, 16)\n", &a);
      fprintf(fp0, "dpim.set(%p, 16)\n", y);
    	fprintf(fp0, "dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
    	fprintf(fp0, "dpim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "dpim.set(%p, 16)\n", &a);
      fprintf(fp0, "dpim.set(%p, 16)\n", y);
    	fprintf(fp0, "dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
    	fprintf(fp0, "dpim.store(%p, 16)\n", y);
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	fprintf(fp0, "dpim.set(%p, 16)\n", &a);
      fprintf(fp0, "dpim.set(%p, 16)\n", y);
    	fprintf(fp0, "dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
    	fprintf(fp0, "dpim.store(%p, 16)\n", y);
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "dpim.set(%p, 16)\n", &a);
      fprintf(fp1, "dpim.set(%p, 16)\n", y);
    	fprintf(fp1, "dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
    	fprintf(fp1, "dpim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "dpim.set(%p, 16)\n", &a);
      fprintf(fp1, "dpim.set(%p, 16)\n", y);
    	fprintf(fp1, "dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
    	fprintf(fp1, "dpim.store(%p, 16)\n", y);
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	fprintf(fp1, "dpim.set(%p, 16)\n", &a);
      fprintf(fp1, "dpim.set(%p, 16)\n", y);
    	fprintf(fp1, "dpim.vector_comp(%p, %p, 16, add)\n", &a, y);
    	fprintf(fp1, "dpim.store(%p, 16)\n", y);
      fclose(fp1);
     }
    #endif  
#endif
#ifdef PNM  
    ADDR(y);     
  #ifdef MAP_TYPE1
  if(LANG_NUM == 0) {
    FILE* fp4 = fopen("./IR/C_SIMD-to-IR_PNM_RecNMP.trace","a"); 
	  fprintf(fp4, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  if(LANG_NUM == 1) {
    FILE* fp4 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_RecNMP.trace","a");
	  fprintf(fp4, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
   if(LANG_NUM == 2) {
    FILE* fp4 = fopen("./IR/CUDA_SIMD-to-IR_PNM_RecNMP.trace","a");
	  fprintf(fp4, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  #endif
  #ifdef MAP_TYPE2  
  if(LANG_NUM == 0) {
    FILE* fp5 = fopen("./IR/C_SIMD-to-IR_PNM_TensorDIMM.trace","a");
	  fprintf(fp5, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 1) {
    FILE* fp5 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace","a");
	  fprintf(fp5, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 2) {
    FILE* fp5 = fopen("./IR/CUDA_SIMD-to-IR_PNM_TensorDIMM.trace","a");
	  fprintf(fp5, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, imm)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
  #endif   
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] + a;
    }
    SIMD_CYCLE++;
    //printf("simd scal add 16\n");
}
/*
void SIMD_SCAL_SUB_16(float a, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector_imm(%f, %p, 16, add)\n", a, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		float imm = a;
		printf("dpim.set(%p, 16)\n", &a);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, sub)\n", &a, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
    ADDR(y);     
	  printf("pnm.vector_imm(a: %f, ", a);  
    printf(" y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, add)\n", ch, ra, bg, ba, ro, co); 
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] - a;
    }
    SIMD_CYCLE++;
    //printf("simd scal sub 16\n");
}*/

void SIMD_SCAL_MUL_16(float a, float* y) {
#ifdef APIM
    #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(&a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector_imm(reg: %d, ", reg); 
    	ADDR(y); 
    	fprintf(fp0, " y: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp0, "apim.store(y: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(&a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector_imm(reg: %d, ", reg); 
    	ADDR(y); 
    	fprintf(fp0, " y: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp0, "apim.store(y: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(&a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector_imm(reg: %d, ", reg); 
    	ADDR(y); 
    	fprintf(fp0, " y: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp0, "apim.store(y: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb); 
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(&a);
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector_imm(reg: %d, ", reg); 
    	ADDR(y);
    	fprintf(fp1, " y: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp1, "apim.store(y: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
      fclose(fp1);
     } 
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(&a);
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector_imm(reg: %d, ", reg); 
    	ADDR(y);
    	fprintf(fp1, " y: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp1, "apim.store(y: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(&a);
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector_imm(reg: %d, ", reg); 
    	ADDR(y);
    	fprintf(fp1, " y: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
    	fprintf(fp1, "apim.store(y: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
      fclose(fp1);
     }
    #endif  
#endif
#ifdef DPIM
  #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp2 = fopen("./IR/C_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(&a);
      fprintf(fp2, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector_comp(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp2, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 1) {
      FILE* fp2 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(&a);
      fprintf(fp2, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector_comp(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp2, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 2) {
      FILE* fp2 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(&a);
      fprintf(fp2, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector_comp(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp2, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
  #endif
  #ifdef MAP_TYPE2   
    if(LANG_NUM == 0) {
      FILE* fp3 = fopen("./IR/C_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(&a);
      fprintf(fp3, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector_comp(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp3, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3);
   }
   if(LANG_NUM == 1) {
      FILE* fp3 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(&a);
      fprintf(fp3, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector_comp(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp3, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3);
   }
   if(LANG_NUM == 2) {
      FILE* fp3 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(&a);
      fprintf(fp3, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector_comp(reg: %d, ", reg); 
		  ADDR(y);
			fprintf(fp3, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
			fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3);
   }
  #endif     
#endif
#ifdef PNM
  #ifdef MAP_TYPE1
  if(LANG_NUM == 0) {
    FILE* fp4 = fopen("./IR/C_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(y);     
	  fprintf(fp4, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  if(LANG_NUM == 1) {
    FILE* fp4 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(y);     
	  fprintf(fp4, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
   if(LANG_NUM == 2) {
    FILE* fp4 = fopen("./IR/CUDA_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(y);     
	  fprintf(fp4, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp4, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  #endif
  #ifdef MAP_TYPE2  
  if(LANG_NUM == 0) {
    FILE* fp5 = fopen("./IR/C_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(y);     
	  fprintf(fp5, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 1) {
    FILE* fp5 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(y);     
	  fprintf(fp5, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
   if(LANG_NUM == 2) {
    FILE* fp5 = fopen("./IR/CUDA_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(y);     
	  fprintf(fp5, "pnm.vector_imm(a: %f, ", a);  
    fprintf(fp5, " y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
    fclose(fp5);
   }
  #endif
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] * a;
    }
    SIMD_CYCLE+=2;
    //printf("simd scal mul 16\n");
}
/*
void SIMD_SCAL_DIV_16(float a, float* y) {
#ifdef APIM
		printf("apim.load(%p, 16)\n", y);
		printf("apim.vector_imm(%f, %p, 16, 3)\n", a, y);
		printf("apim.store(%p, 16)\n", y);
#endif
#ifdef DPIM
		float imm = a;
		printf("dpim.set(%p, 16)\n", &a);
		printf("dpim.set(%p, 16)\n", y);
		printf("dpim.vector_comp(%p, %p, 16, div)\n", &a, y);
		printf("dpim.store(%p, 16)\n", y);
#endif
#ifdef PNM
    ADDR(y);     
	  printf("pnm.vector_imm(a: %f, ", a);  
    printf(" y: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, 3)\n", ch, ra, bg, ba, ro, co);  
#endif
    for(int i=0; i<16; i++) {
        y[i] = y[i] / a;
    }
    SIMD_CYCLE+=4;
    //printf("simd scal div 16\n");
}*/


float MAC_16(float* v_a, float* v_b) {
#ifdef APIM
      #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(v_a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector(reg: %d, ", reg); 
    	ADDR(v_b); 
    	fprintf(fp0, " b: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
     fprintf(fp0, "apim.vector_acc(b: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp0, "apim.store(b: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(v_a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector(reg: %d, ", reg); 
    	ADDR(v_b); 
    	fprintf(fp0, " b: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
     fprintf(fp0, "apim.vector_acc(b: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp0, "apim.store(b: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(v_a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp0, "apim.vector(reg: %d, ", reg); 
    	ADDR(v_b); 
    	fprintf(fp0, " b: tile %d, mac_unit %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
     fprintf(fp0, "apim.vector_acc(b: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp0, "apim.store(b: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(v_a); 
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector(reg: %d, ", reg); 
    	ADDR(v_b); 
    	fprintf(fp1, " b: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
     fprintf(fp1, "apim.vector_acc(b: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp1, "apim.store(b: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp1);
     }
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(v_a); 
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector(reg: %d, ", reg); 
    	ADDR(v_b); 
    	fprintf(fp1, " b: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
     fprintf(fp1, "apim.vector_acc(b: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp1, "apim.store(b: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(v_a); 
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
    	fprintf(fp1, "apim.vector(reg: %d, ", reg); 
    	ADDR(v_b); 
    	fprintf(fp1, " b: chip %d, reram_bank %d, x-bar %d, add, 16, mul)\n", ti, rb, xb); 
     fprintf(fp1, "apim.vector_acc(b: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp1, "apim.store(b: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp1);
     }
    #endif 
#endif

#ifdef DPIM  
     #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp2 = fopen("./IR/C_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(v_a);
      fprintf(fp2, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(v_b);
      fprintf(fp2, "b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 1) {
      FILE* fp2 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(v_a);
      fprintf(fp2, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(v_b);
      fprintf(fp2, "b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 2) {
      FILE* fp2 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(v_a);
      fprintf(fp2, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(v_b);
      fprintf(fp2, "b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
  #endif
  #ifdef MAP_TYPE2   
    if(LANG_NUM == 0) {
      FILE* fp3 = fopen("./IR/C_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(v_a);
      fprintf(fp3, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(v_b);
      fprintf(fp3, "b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3); 
   }
   if(LANG_NUM == 1) {
      FILE* fp3 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(v_a);
      fprintf(fp3, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(v_b);
      fprintf(fp3, "b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3); 
   }
   if(LANG_NUM == 2) {
      FILE* fp3 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(v_a);
      fprintf(fp3, "dpim.load(a: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, add, 16)\n", ch, ra, bg, ba, ro, co, reg);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(v_b);
      fprintf(fp3, "b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3); 
   }
  #endif  
#endif
#ifdef PNM  
    #ifdef MAP_TYPE1
  if(LANG_NUM == 0) {
    FILE* fp4 = fopen("./IR/C_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(v_a);   
	  fprintf(fp4, "pnm.vector_op(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(v_b); 
    fprintf(fp4, " b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co); 
    fclose(fp4);
   }
  if(LANG_NUM == 1) {
    FILE* fp4 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(v_a);   
	  fprintf(fp4, "pnm.vector_op(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(v_b); 
    fprintf(fp4, " b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co); 
    fclose(fp4);
   }
   if(LANG_NUM == 2) {
    FILE* fp4 = fopen("./IR/CUDA_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(v_a);   
	  fprintf(fp4, "pnm.vector_op(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(v_b); 
    fprintf(fp4, " b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co); 
    fclose(fp4);
   }
  #endif
  #ifdef MAP_TYPE2  
  if(LANG_NUM == 0) {
    FILE* fp5 = fopen("./IR/C_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(v_a);   
	  fprintf(fp5, "pnm.vector_op(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(v_b); 
    fprintf(fp5, " b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co); 
    fclose(fp5);
   }
   if(LANG_NUM == 1) {
    FILE* fp5 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(v_a);   
	  fprintf(fp5, "pnm.vector_op(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(v_b); 
    fprintf(fp5, " b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co); 
    fclose(fp5);
   }
   if(LANG_NUM == 2) {
    FILE* fp5 = fopen("./IR/CUDA_SIMD-to-IR_PNM_TensorDIMM.trace","a");
     ADDR(v_a);   
	  fprintf(fp5, "pnm.vector_op(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    ADDR(v_b); 
    fprintf(fp5, " b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16, mul)\n", ch, ra, bg, ba, ro, co); 
    fclose(fp5);
   }
  #endif 
#endif
    float rt = 0;
    for(int i=0; i<16; i++) {
        rt += v_a[i] * v_b[i];
    }
    SIMD_CYCLE+=3;
    //printf("mac 16\n");
    return rt;
}

float ACC_16(float* v_a) {
#ifdef APIM 
 #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp0 = fopen("./IR/C_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(v_a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
     fprintf(fp0, "apim.vector_acc(b: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp0, "apim.store(b: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 1) {
      FILE* fp0 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(v_a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
     fprintf(fp0, "apim.vector_acc(b: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp0, "apim.store(b: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp0);
     }
     if(LANG_NUM == 2) {
      FILE* fp0 = fopen("./IR/CUDA_SIMD-to-IR_APIM_ISSAC.trace","a");
    	ADDR(v_a); 
    	fprintf(fp0, "apim.load(a: tile %d, mac_unit %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
     fprintf(fp0, "apim.vector_acc(b: tile %d, mac_unit %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp0, "apim.store(b: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp0);
     }
    #endif
    #ifdef MAP_TYPE2
    if(LANG_NUM == 0) {
      FILE* fp1 = fopen("./IR/C_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(v_a); 
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
     fprintf(fp1, "apim.vector_acc(b: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp1, "apim.store(b: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp1);
     }
     if(LANG_NUM == 1) {
      FILE* fp1 = fopen("./IR/OpenMP_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(v_a); 
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
     fprintf(fp1, "apim.vector_acc(b: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp1, "apim.store(b: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp1);
     } 
     if(LANG_NUM == 2) {
      FILE* fp1 = fopen("./IR/CUDA_SIMD-to-IR_APIM_PRIME.trace","a");
    	ADDR(v_a); 
    	fprintf(fp1, "apim.load(a: chip %d, reram_bank %d, x-bar %d, reg: %d, add, 16)\n", ti, rb, xb, reg); 
     fprintf(fp1, "apim.vector_acc(b: chip %d, reram_bank %d, x-bar %d, add, 16)\n", ti, rb, xb);
    	fprintf(fp1, "apim.store(b: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
      fclose(fp1);
     }
    #endif 
#endif
#ifdef DPIM   
  #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp2 = fopen("./IR/C_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(v_a);
		  fprintf(fp2, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 1) {
      FILE* fp2 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(v_a);
		  fprintf(fp2, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
     if(LANG_NUM == 2) {
      FILE* fp2 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_Newton.trace","a");
		  ADDR(v_a);
		  fprintf(fp2, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp2); 
     }
  #endif
  #ifdef MAP_TYPE2   
    if(LANG_NUM == 0) {
      FILE* fp3 = fopen("./IR/C_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(v_a);
		  fprintf(fp3, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3); 
   }
   if(LANG_NUM == 1) {
      FILE* fp3 = fopen("./IR/OpenMP_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(v_a);
		  fprintf(fp3, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3); 
   }
   if(LANG_NUM == 2) {
      FILE* fp3 = fopen("./IR/CUDA_SIMD-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(v_a);
		  fprintf(fp3, "dpim.vector_acc(b: ch %d ra %d bg %d ba %d ro %d co %d, add, 16)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(b: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co);
      fclose(fp3); 
   }
  #endif 
#endif
#ifdef PNM     
     #ifdef MAP_TYPE1
  if(LANG_NUM == 0) {
    FILE* fp4 = fopen("./IR/C_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(v_a);   
	  fprintf(fp4, "pnm.vector_acc(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
  if(LANG_NUM == 1) {
    FILE* fp4 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(v_a);   
	  fprintf(fp4, "pnm.vector_acc(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
    fclose(fp4);
   }
   if(LANG_NUM == 2) {
    FILE* fp4 = fopen("./IR/CUDA_SIMD-to-IR_PNM_RecNMP.trace","a");
    ADDR(v_a);   
	  fprintf(fp4, "pnm.vector_acc(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
    fclose(fp4);
   }
  #endif
  #ifdef MAP_TYPE2  
  if(LANG_NUM == 0) {
    FILE* fp5 = fopen("./IR/C_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(v_a);   
	  fprintf(fp5, "pnm.vector_acc(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
    fclose(fp5);
   }
   if(LANG_NUM == 1) {
    FILE* fp5 = fopen("./IR/OpenMP_SIMD-to-IR_PNM_TensorDIMM.trace","a");
    ADDR(v_a);   
	  fprintf(fp5, "pnm.vector_acc(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
    fclose(fp5);
   }
   if(LANG_NUM == 2) {
    FILE* fp5 = fopen("./IR/CUDA_SIMD-to-IR_PNM_TensorDIMM.trace","a");
     ADDR(v_a);   
	  fprintf(fp5, "pnm.vector_acc(a: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co); 
    fclose(fp5);
   }
  #endif   
#endif
    float rt = 0;
    for(int i=0; i<16; i++) {
        rt += v_a[i];
    }
    SIMD_CYCLE+=2;
    //printf("acc 16\n");
    return rt;
}

void USE_SHARED_MEM() {
    assert(0);
}

#endif 

#ifdef BLAS // matrix&vector level operations (simple blas)
  float VEC_ACC(float *x, int width) {
	int temp = width;
  #ifdef APIM
	  while(temp>1){
  		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
  		#ifdef MAP_TYPE1
        if(LANG_NUM == 0) {
          FILE* fp6 = fopen("./IR/C_Func-to-IR_APIM_ISSAC.trace","a");
          ADDR(x+i);
  				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
          fprintf(fp6, "apim.vector_acc(a: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
          fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
          fclose(fp6); 
        }
        if(LANG_NUM == 1) {
          FILE* fp6 = fopen("./IR/OpenMP_Func-to-IR_APIM_ISSAC.trace","a");
          ADDR(x+i);
  				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
          fprintf(fp6, "apim.vector_acc(a: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
          fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
          fclose(fp6); 
        }
        if(LANG_NUM == 2) {
          FILE* fp6 = fopen("./IR/CUDA_Func-to-IR_APIM_ISSAC.trace","a");
          ADDR(x+i);
  				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
          fprintf(fp6, "apim.vector_acc(a: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
          fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, 1)\n", ti, rb, xb); 
          fclose(fp6); 
        }                    
  	   #endif 
  	   #ifdef MAP_TYPE2
        if(LANG_NUM == 0) {
          FILE* fp7 = fopen("./IR/C_Func-to-IR_APIM_PRIME.trace","a");
          ADDR(x+i);
  				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
          fprintf(fp7, "apim.vector_acc(a: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
          fprintf(fp7, "apim.store(x: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb); 
          fclose(fp7);
        }	
        if(LANG_NUM == 1) {
          FILE* fp7 = fopen("./IR/OpenMP_Func-to-IR_APIM_PRIME.trace","a");
          ADDR(x+i);
  				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
          fprintf(fp7, "apim.vector_acc(a: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
          fprintf(fp7, "apim.store(x: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb);
          fclose(fp7);
        }		
        if(LANG_NUM == 2) {
          FILE* fp7 = fopen("./IR/CUDA_Func-to-IR_APIM_PRIME.trace","a");
          ADDR(x+i);
  				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
          fprintf(fp7, "apim.vector_acc(a: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
          fprintf(fp7, "apim.store(x: chip %d, reram_bank %d, x-bar %d, 1)\n", ti, rb, xb);
          fclose(fp7);
        }	
    	 	 #endif
        }
    		temp/=APIM_SIMD_SIZE;
    	}
     temp = width;
   #endif
  
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE) {          
      #ifdef MAP_TYPE1
        if(LANG_NUM == 0) {
          FILE* fp2 = fopen("./IR/C_Func-to-IR_DPIM_Newton.trace","a");
    		  ADDR(x+i);
    	    fprintf(fp2, "dpim.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          ADDR(x);
	        fprintf(fp2, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co); 
          fclose(fp2); 
         }
         if(LANG_NUM == 1) {
          FILE* fp2 = fopen("./IR/OpenMP_Func-to-IR_DPIM_Newton.trace","a");
    		  ADDR(x+i);
    	    fprintf(fp2, "dpim.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          ADDR(x);
	        fprintf(fp2, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co); 
          fclose(fp2); 
         }
         if(LANG_NUM == 2) {
          FILE* fp2 = fopen("./IR/CUDA_Func-to-IR_DPIM_Newton.trace","a");
    		  ADDR(x+i);
    	    fprintf(fp2, "dpim.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          ADDR(x);
	        fprintf(fp2, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co); 
          fclose(fp2); 
         }
      #endif
      #ifdef MAP_TYPE2   
        if(LANG_NUM == 0) {
          FILE* fp3 = fopen("./IR/C_Func-to-IR_DPIM_HBM-PIM.trace","a");
    		  ADDR(x+i);
    	    fprintf(fp3, "dpim.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          ADDR(x);
	        fprintf(fp3, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co); 
          fclose(fp3); 
       }
       if(LANG_NUM == 1) {
          FILE* fp3 = fopen("./IR/OpenMP_Func-to-IR_DPIM_HBM-PIM.trace","a");
    		  ADDR(x+i);
    	    fprintf(fp3, "dpim.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          ADDR(x);
	        fprintf(fp3, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co); 
          fclose(fp3); 
       }
       if(LANG_NUM == 2) {
          FILE* fp3 = fopen("./IR/CUDA_Func-to-IR_DPIM_HBM-PIM.trace","a");
    		  ADDR(x+i);
    	    fprintf(fp3, "dpim.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          ADDR(x);
	        fprintf(fp3, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, 1)\n", ch, ra, bg, ba, ro, co); 
          fclose(fp3); 
       } 
       #endif       
		}
		temp/=DPIM_SIMD_SIZE;
	}
 temp = width;
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
      ADDR(x+i);              
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp10 = fopen("./IR/C_Func-to-IR_PNM_RecNMP.trace","a");
        fprintf(fp10, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 1) {
        FILE* fp10 = fopen("./IR/OpenMP_Func-to-IR_PNM_RecNMP.trace","a");
        fprintf(fp10, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 2) {
        FILE* fp10 = fopen("./IR/CUDA_Func-to-IR_PNM_RecNMP.trace","a");
        fprintf(fp10, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp11 = fopen("./IR/C_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        fprintf(fp11, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 1) {
        FILE* fp11 = fopen("./IR/OpenMP_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        fprintf(fp11, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 2) {
        FILE* fp11 = fopen("./IR/CUDA_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        fprintf(fp11, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
 			#endif          
		}
		temp/=PNM_SIMD_SIZE;
	}
 temp = width;
#endif
	float rt = 0;
	for(int i=0; i<width; i++) {
		rt += x[i];
	}
  MV_CYCLE++;
  //printf("acc\n");
	return rt;
}

void VEC_ADD(float *x, float *y, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp6 = fopen("./IR/C_Func-to-IR_APIM_ISSAC.trace","a");
        ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
        fprintf(fp6, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp6, " y: tile %d, mac_unit %d, x-bar %d, %d, imm)\n", ti, rb, xb, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 
        fclose(fp6); 
      }
      if(LANG_NUM == 1) {
        FILE* fp6 = fopen("./IR/OpenMP_Func-to-IR_APIM_ISSAC.trace","a");
        ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
        fprintf(fp6, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp6, " y: tile %d, mac_unit %d, x-bar %d, %d, imm)\n", ti, rb, xb, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 
        fclose(fp6); 
      }
      if(LANG_NUM == 2) {
        FILE* fp6 = fopen("./IR/CUDA_Func-to-IR_APIM_ISSAC.trace","a");
        ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
        fprintf(fp6, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp6, " y: tile %d, mac_unit %d, x-bar %d, %d, imm)\n", ti, rb, xb, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 
        fclose(fp6); 
      }
			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp7 = fopen("./IR/C_Func-to-IR_APIM_PRIME.trace","a");
        ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
        fprintf(fp7, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp7, " y: tile %d, mac_unit %d, x-bar %d, %d, imm)\n", ti, rb, xb, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 
        fclose(fp7);
      }	
      if(LANG_NUM == 1) {
        FILE* fp7 = fopen("./IR/OpenMP_Func-to-IR_APIM_PRIME.trace","a");
        ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
        fprintf(fp7, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp7, " y: tile %d, mac_unit %d, x-bar %d, %d, imm)\n", ti, rb, xb, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 
        fclose(fp7);
      }		
      if(LANG_NUM == 2) {
        FILE* fp7 = fopen("./IR/CUDA_Func-to-IR_APIM_PRIME.trace","a");
        ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg, APIM_SIMD_SIZE); 
        fprintf(fp7, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp7, " y: tile %d, mac_unit %d, x-bar %d, %d, imm)\n", ti, rb, xb, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 
        fclose(fp7);
      }	
			#endif        
		}
		temp/=APIM_SIMD_SIZE;
	}
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){  
      #ifdef MAP_TYPE1
        if(LANG_NUM == 0) {
          FILE* fp2 = fopen("./IR/C_Func-to-IR_DPIM_Newton.trace","a");
    		  ADDR(x);
          fprintf(fp2, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
    	    fprintf(fp2, "dpim.vector_comp(reg: %d, ", reg); 
    		  ADDR(y+i);
          fprintf(fp2, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, add)\n", ch, ra, bg, ba, ro, co);
    		  fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          fclose(fp2); 
         }
         if(LANG_NUM == 1) {
          FILE* fp2 = fopen("./IR/OpenMP_Func-to-IR_DPIM_Newton.trace","a");
    		  ADDR(x);
          fprintf(fp2, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
    	    fprintf(fp2, "dpim.vector_comp(reg: %d, ", reg); 
    		  ADDR(y+i);
          fprintf(fp2, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, add)\n", ch, ra, bg, ba, ro, co);
    		  fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          fclose(fp2); 
         }
         if(LANG_NUM == 2) {
          FILE* fp2 = fopen("./IR/CUDA_Func-to-IR_DPIM_Newton.trace","a");
    		  ADDR(x);
          fprintf(fp2, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
    	    fprintf(fp2, "dpim.vector_comp(reg: %d, ", reg); 
    		  ADDR(y+i);
          fprintf(fp2, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, add)\n", ch, ra, bg, ba, ro, co);
    		  fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          fclose(fp2); 
         }
      #endif
      #ifdef MAP_TYPE2   
        if(LANG_NUM == 0) {
          FILE* fp3 = fopen("./IR/C_Func-to-IR_DPIM_HBM-PIM.trace","a");
    		  ADDR(x);
          fprintf(fp3, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
    	    fprintf(fp3, "dpim.vector_comp(reg: %d, ", reg); 
    		  ADDR(y+i);
          fprintf(fp3, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, add)\n", ch, ra, bg, ba, ro, co);
    		  fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          fclose(fp3); 
       }
       if(LANG_NUM == 1) {
          FILE* fp3 = fopen("./IR/OpenMP_Func-to-IR_DPIM_HBM-PIM.trace","a");
    		  ADDR(x);
          fprintf(fp3, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
    	    fprintf(fp3, "dpim.vector_comp(reg: %d, ", reg); 
    		  ADDR(y+i);
          fprintf(fp3, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, add)\n", ch, ra, bg, ba, ro, co);
    		  fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          fclose(fp3); 
       }
       if(LANG_NUM == 2) {
          FILE* fp3 = fopen("./IR/CUDA_Func-to-IR_DPIM_HBM-PIM.trace","a");
    		  ADDR(x);
          fprintf(fp3, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
    	    fprintf(fp3, "dpim.vector_comp(reg: %d, ", reg); 
    		  ADDR(y+i);
          fprintf(fp3, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, add)\n", ch, ra, bg, ba, ro, co);
    		  fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
          fclose(fp3); 
       } 
       #endif   
		}
		temp/=DPIM_SIMD_SIZE;
	}
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){      
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp10 = fopen("./IR/C_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp10, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, imm)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 1) {
        FILE* fp10 = fopen("./IR/OpenMP_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp10, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, imm)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 2) {
        FILE* fp10 = fopen("./IR/CUDA_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp10, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, imm)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp11 = fopen("./IR/C_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp11, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, imm)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 1) {
        FILE* fp11 = fopen("./IR/OpenMP_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp11, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, imm)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 2) {
        FILE* fp11 = fopen("./IR/CUDA_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_acc(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp11, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, imm)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
 			#endif         
		}
		temp/=PNM_SIMD_SIZE;
	}
#endif
	for(int i=0; i<width; i++) {
		y[i] += x[i];
	}
  MV_CYCLE++;
  //printf("add\n");
}
void VEC_MUL(float *x, float *y, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){    
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp6 = fopen("./IR/C_Func-to-IR_APIM_ISSAC.trace","a");
        ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE); 
        fprintf(fp6, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp6, " y: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", ti, rb, xb,APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE); 
        fclose(fp6); 
      }
      if(LANG_NUM == 1) {
        FILE* fp6 = fopen("./IR/OpenMP_Func-to-IR_APIM_ISSAC.trace","a");
        ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE); 
        fprintf(fp6, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp6, " y: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", ti, rb, xb,APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE); 
        fclose(fp6); 
      }
      if(LANG_NUM == 2) {
        FILE* fp6 = fopen("./IR/CUDA_Func-to-IR_APIM_ISSAC.trace","a");
        ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE); 
        fprintf(fp6, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp6, " y: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", ti, rb, xb,APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE); 
        fclose(fp6); 
      }
			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp7 = fopen("./IR/C_Func-to-IR_APIM_PRIME.trace","a");
        ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE); 
        fprintf(fp7, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp7, " y: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", ti, rb, xb,APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE); 
        fclose(fp7);
      }	
      if(LANG_NUM == 1) {
        FILE* fp7 = fopen("./IR/OpenMP_Func-to-IR_APIM_PRIME.trace","a");
        ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE); 
        fprintf(fp7, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp7, " y: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", ti, rb, xb,APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE); 
        fclose(fp7);
      }		
      if(LANG_NUM == 2) {
        FILE* fp7 = fopen("./IR/CUDA_Func-to-IR_APIM_PRIME.trace","a");
        ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE); 
        fprintf(fp7, "apim.vector(reg: %d, ", reg); 
        ADDR(y+i);
				fprintf(fp7, " y: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", ti, rb, xb,APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE); 
        fclose(fp7);
      }	
			#endif    
		}
		temp/=APIM_SIMD_SIZE;
	}
  temp = width;
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
    #ifdef MAP_TYPE1
    if(LANG_NUM == 0) {
      FILE* fp2 = fopen("./IR/C_Func-to-IR_DPIM_Newton.trace","a");
		  ADDR(x+i);
      fprintf(fp2, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y+i);
      fprintf(fp2, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
      fclose(fp2); 
     }
     if(LANG_NUM == 1) {
      FILE* fp2 = fopen("./IR/OpenMP_Func-to-IR_DPIM_Newton.trace","a");
		  ADDR(x+i);
      fprintf(fp2, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y+i);
      fprintf(fp2, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
      fclose(fp2); 
     }
     if(LANG_NUM == 2) {
      FILE* fp2 = fopen("./IR/CUDA_Func-to-IR_DPIM_Newton.trace","a");
		  ADDR(x+i);
      fprintf(fp2, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
	    fprintf(fp2, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y+i);
      fprintf(fp2, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp2, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
      fclose(fp2); 
     }
  #endif
  #ifdef MAP_TYPE2   
    if(LANG_NUM == 0) {
      FILE* fp3 = fopen("./IR/C_Func-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(x+i);
      fprintf(fp3, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y+i);
      fprintf(fp3, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
      fclose(fp3); 
   }
   if(LANG_NUM == 1) {
      FILE* fp3 = fopen("./IR/OpenMP_Func-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(x+i);
      fprintf(fp3, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y+i);
      fprintf(fp3, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
      fclose(fp3); 
   }
   if(LANG_NUM == 2) {
      FILE* fp3 = fopen("./IR/CUDA_Func-to-IR_DPIM_HBM-PIM.trace","a");
		  ADDR(x+i);
      fprintf(fp3, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg, DPIM_SIMD_SIZE);
	    fprintf(fp3, "dpim.vector(reg: %d, ", reg); 
		  ADDR(y+i);
      fprintf(fp3, "y: ch %d ra %d bg %d ba %d ro %d co %d, 16, mul)\n", ch, ra, bg, ba, ro, co);
		  fprintf(fp3, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);
      fclose(fp3); 
   }
  #endif    
		}
		temp/=DPIM_SIMD_SIZE;
	}
 temp = width;
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp10 = fopen("./IR/C_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp10, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 1) {
        FILE* fp10 = fopen("./IR/OpenMP_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp10, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 2) {
        FILE* fp10 = fopen("./IR/CUDA_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp10, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp11 = fopen("./IR/C_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp11, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 1) {
        FILE* fp11 = fopen("./IR/OpenMP_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp11, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 2) {
        FILE* fp11 = fopen("./IR/CUDA_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector(x: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(y+i);
        fprintf(fp11, " y: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
 			#endif     
		}
		temp/=PNM_SIMD_SIZE;
	}
 temp = width;
#endif
	for(int i=0; i<width; i++) {
		y[i] *= x[i];
	}
  MV_CYCLE+=2;
  //printf("mul\n");
}
void VEC_IMM_ADD(float imm, float *x, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp6 = fopen("./IR/C_Func-to-IR_APIM_ISSAC.trace","a");
				fprintf(fp6, "apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.vector_imm(%f, %p, %d, imm)\n", imm, x+i, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);	
        fclose(fp6); 
      }
      if(LANG_NUM == 1) {
        FILE* fp6 = fopen("./IR/OpenMP_Func-to-IR_APIM_ISSAC.trace","a");
				fprintf(fp6, "apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.vector_imm(%f, %p, %d, imm)\n", imm, x+i, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);	
        fclose(fp6); 
      }
      if(LANG_NUM == 2) {
        FILE* fp6 = fopen("./IR/CUDA_Func-to-IR_APIM_ISSAC.trace","a");
				fprintf(fp6, "apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.vector_imm(%f, %p, %d, imm)\n", imm, x+i, APIM_SIMD_SIZE);
				fprintf(fp6, "apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);		
        fclose(fp6); 
      }
			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp7 = fopen("./IR/C_Func-to-IR_APIM_PRIME.trace","a");
				fprintf(fp7, "apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.vector_imm(%f, %p, %d, imm)\n", imm, x+i, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);
        fclose(fp7);
      }	
      if(LANG_NUM == 1) {
        FILE* fp7 = fopen("./IR/OpenMP_Func-to-IR_APIM_PRIME.trace","a");
				fprintf(fp7, "apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.vector_imm(%f, %p, %d, imm)\n", imm, x+i, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);
        fclose(fp7);
      }		
      if(LANG_NUM == 2) {
        FILE* fp7 = fopen("./IR/CUDA_Func-to-IR_APIM_PRIME.trace","a");
				fprintf(fp7, "apim.load(%p, %d)\n", x+i, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.vector_imm(%f, %p, %d, imm)\n", imm, x+i, APIM_SIMD_SIZE);
				fprintf(fp7, "apim.store(%p, %d)\n", x+i, APIM_SIMD_SIZE);
        fclose(fp7);
      }	
			#endif    
		}
		temp/=APIM_SIMD_SIZE;
	}
 temp = width;
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=DPIM_SIMD_SIZE){
    #ifdef MAP_TYPE1
         if(LANG_NUM == 0) {
            FILE* fp8 = fopen("./IR/C_Func-to-IR_DPIM_Newton.trace","a"); 
      		  fprintf(fp8, "dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
            fprintf(fp8, "dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
      		  fprintf(fp8, "dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
    				fprintf(fp8, "dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
            fclose(fp8);
          } 
         if(LANG_NUM == 1) {
            FILE* fp8 = fopen("./IR/OpenMP_Func-to-IR_DPIM_Newton.trace","a"); 
      		  fprintf(fp8, "dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
            fprintf(fp8, "dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
      		  fprintf(fp8, "dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
    				fprintf(fp8, "dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
            fclose(fp8);
          } 
          if(LANG_NUM == 2) {
            FILE* fp8 = fopen("./IR/CUDA_Func-to-IR_DPIM_Newton.trace","a"); 
      		  fprintf(fp8, "dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
            fprintf(fp8, "dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
      		  fprintf(fp8, "dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
    				fprintf(fp8, "dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
            fclose(fp8);
          } 
     			#endif
    			#ifdef MAP_TYPE2
          if(LANG_NUM == 0) {
            FILE* fp9 = fopen("./IR/C_Func-to-IR_DPIM_HBM-PIM.trace","a");
      		  fprintf(fp9, "dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
            fprintf(fp9, "dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
      		  fprintf(fp9, "dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
    				fprintf(fp9, "dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
            fclose(fp9);
          } 
          if(LANG_NUM == 1) {
            FILE* fp9 = fopen("./IR/OpenMP_Func-to-IR_DPIM_HBM-PIM.trace","a");
      		  fprintf(fp9, "dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
            fprintf(fp9, "dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
      		  fprintf(fp9, "dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
    				fprintf(fp9, "dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
            fclose(fp9);
          }
          if(LANG_NUM == 2) {
            FILE* fp9 = fopen("./IR/CUDA_Func-to-IR_DPIM_HBM-PIM.trace","a");
      		  fprintf(fp9, "dpim.set(%p, %d)\n", &imm, DPIM_SIMD_SIZE);
            fprintf(fp9, "dpim.set(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
      		  fprintf(fp9, "dpim.vector_comp(%p, %p, %d, add)\n", &imm, x+i, DPIM_SIMD_SIZE);
    				fprintf(fp9, "dpim.store(%p, %d)\n", x+i, DPIM_SIMD_SIZE);
            fclose(fp9);
          }
     			#endif   
		}
		temp/=DPIM_SIMD_SIZE;
	}
 temp = width;
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){     
    #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp10 = fopen("./IR/C_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 1) {
        FILE* fp10 = fopen("./IR/OpenMP_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 2) {
        FILE* fp10 = fopen("./IR/CUDA_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp11 = fopen("./IR/C_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 1) {
        FILE* fp11 = fopen("./IR/OpenMP_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 2) {
        FILE* fp11 = fopen("./IR/CUDA_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
 			#endif 
		}
		temp/=PNM_SIMD_SIZE;
	}
 temp = width;
#endif
	for(int i=0; i<width; i++) {
		x[i] += imm;
	}
  MV_CYCLE++;
  //printf("imm add\n");
}
void VEC_IMM_MUL(float imm, float *x, int width) {
	int temp = width;
#ifdef APIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
			#ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp6 = fopen("./IR/C_Func-to-IR_APIM_ISSAC.trace","a");
			  ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
				fprintf(fp6, "apim.vector(imm: %f, x: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", imm, ti, rb, xb, APIM_SIMD_SIZE);	
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 		
        fclose(fp6); 
      }
      if(LANG_NUM == 1) {
        FILE* fp6 = fopen("./IR/OpenMP_Func-to-IR_APIM_ISSAC.trace","a");
			  ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
				fprintf(fp6, "apim.vector(imm: %f, x: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", imm, ti, rb, xb, APIM_SIMD_SIZE);	
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 		
        fclose(fp6); 
      }
      if(LANG_NUM == 2) {
        FILE* fp6 = fopen("./IR/CUDA_Func-to-IR_APIM_ISSAC.trace","a");
			  ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
				fprintf(fp6, "apim.vector(imm: %f, x: tile %d, mac_unit %d, x-bar %d, %d, mul)\n", imm, ti, rb, xb, APIM_SIMD_SIZE);	
				fprintf(fp6, "apim.store(x: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 		
        fclose(fp6); 
      }
			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp7 = fopen("./IR/C_Func-to-IR_APIM_PRIME.trace","a");
 			  ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
				fprintf(fp7, "apim.vector(imm: %f, x: chip %d, reram_bank %d, x-bar %d, %d, mul)\n", imm, ti, rb, xb, APIM_SIMD_SIZE); 
	  	  fprintf(fp7, "apim.store(x: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE);
        fclose(fp7);
      }	
      if(LANG_NUM == 1) {
        FILE* fp7 = fopen("./IR/OpenMP_Func-to-IR_APIM_PRIME.trace","a");
 			  ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
				fprintf(fp7, "apim.vector(imm: %f, x: chip %d, reram_bank %d, x-bar %d, %d, mul)\n", imm, ti, rb, xb, APIM_SIMD_SIZE); 
	  	  fprintf(fp7, "apim.store(x: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE);
        fclose(fp7);
      }		
      if(LANG_NUM == 2) {
        FILE* fp7 = fopen("./IR/CUDA_Func-to-IR_APIM_PRIME.trace","a");
 			  ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
				fprintf(fp7, "apim.vector(imm: %f, x: chip %d, reram_bank %d, x-bar %d, %d, mul)\n", imm, ti, rb, xb, APIM_SIMD_SIZE); 
	  	  fprintf(fp7, "apim.store(x: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb,APIM_SIMD_SIZE);
        fclose(fp7);
      }	
			#endif
		}
		temp/=APIM_SIMD_SIZE;
	}
  temp = width;
#endif
#ifdef DPIM
	while(temp>1){
		for(int i=0; i<temp; i+=APIM_SIMD_SIZE){
     #ifdef MAP_TYPE1
     if(LANG_NUM == 0) {
        FILE* fp8 = fopen("./IR/C_Func-to-IR_DPIM_Newton.trace","a"); 
  	    ADDR(x+i);
  		  fprintf(fp8, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp8, "dpim.vector(imm: %f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp8, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp8);
      } 
     if(LANG_NUM == 1) {
        FILE* fp8 = fopen("./IR/OpenMP_Func-to-IR_DPIM_Newton.trace","a"); 
  	    ADDR(x+i);
  		  fprintf(fp8, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp8, "dpim.vector(imm: %f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp8, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp8);
      } 
      if(LANG_NUM == 2) {
        FILE* fp8 = fopen("./IR/CUDA_Func-to-IR_DPIM_Newton.trace","a"); 
  	    ADDR(x+i);
  		  fprintf(fp8, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp8, "dpim.vector(imm: %f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp8, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp8);
      } 
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp9 = fopen("./IR/C_Func-to-IR_DPIM_HBM-PIM.trace","a");
  	    ADDR(x+i);
  		  fprintf(fp9, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp9, "dpim.vector(imm: %f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp9, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE); 
        fclose(fp9);
      } 
      if(LANG_NUM == 1) {
        FILE* fp9 = fopen("./IR/OpenMP_Func-to-IR_DPIM_HBM-PIM.trace","a");
  	    ADDR(x+i);
  		  fprintf(fp9, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp9, "dpim.vector(imm: %f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp9, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE); 
        fclose(fp9);
      }
      if(LANG_NUM == 2) {
        FILE* fp9 = fopen("./IR/CUDA_Func-to-IR_DPIM_HBM-PIM.trace","a");
  	    ADDR(x+i);
  		  fprintf(fp9, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp9, "dpim.vector(imm: %f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp9, "dpim.store(x: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE); 
        fclose(fp9);
      }
 			#endif
		}
		temp/=DPIM_SIMD_SIZE;
	}
  temp = width;
#endif
#ifdef PNM
	while(temp>1){
		for(int i=0; i<temp; i+=PNM_SIMD_SIZE){
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp10 = fopen("./IR/C_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 1) {
        FILE* fp10 = fopen("./IR/OpenMP_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 2) {
        FILE* fp10 = fopen("./IR/CUDA_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(x+i);
        fprintf(fp10, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp11 = fopen("./IR/C_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 1) {
        FILE* fp11 = fopen("./IR/OpenMP_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 2) {
        FILE* fp11 = fopen("./IR/CUDA_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(x+i);
        fprintf(fp11, "pnm.vector_imm(%f, x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", imm, ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
 			#endif
		}
		temp/=PNM_SIMD_SIZE;
	}
  temp = width;
#endif
	for(int i=0; i<width; i++) {
		x[i] *= imm;
	}
  MV_CYCLE++;
  //printf("imm mul\n");
}
void MV_MUL(float *x, float *A, float *y, int N, int M) {
#ifdef APIM
	int n1 = N/APIM_SIMD_SIZE;
	int m1 = M/APIM_SIMD_SIZE;
	for(int i=0; i<n1; i++){
		for(int j=0; j<m1; j++){   
			#ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp6 = fopen("./IR/C_Func-to-IR_APIM_ISSAC.trace","a");
			  ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
        ADDR(A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
				fprintf(fp6, "apim.xbar_init(A: tile %d, mac_unit %d, x-bar %d)\n", ti, rb, xb);
        ADDR(x+i*APIM_SIMD_SIZE);
        fprintf(fp6, "apim.mvmul(x: tile %d, mac_unit %d, x-bar %d)\n", ti, rb, xb);
        ADDR(y+i*APIM_SIMD_SIZE+j);
				fprintf(fp6, "apim.store(y: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 		
        fclose(fp6); 
      }
      if(LANG_NUM == 1) {
        FILE* fp6 = fopen("./IR/OpenMP_Func-to-IR_APIM_ISSAC.trace","a");
			  ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
        ADDR(A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
				fprintf(fp6, "apim.xbar_init(A: tile %d, mac_unit %d, x-bar %d)\n", ti, rb, xb);
        ADDR(x+i*APIM_SIMD_SIZE);
        fprintf(fp6, "apim.mvmul(x: tile %d, mac_unit %d, x-bar %d)\n", ti, rb, xb);
        ADDR(y+i*APIM_SIMD_SIZE+j);
				fprintf(fp6, "apim.store(y: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 		
        fclose(fp6); 
      }
      if(LANG_NUM == 2) {
        FILE* fp6 = fopen("./IR/CUDA_Func-to-IR_APIM_ISSAC.trace","a");
			  ADDR(x+i);
				fprintf(fp6, "apim.load(x: tile %d, mac_unit %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
        ADDR(A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
				fprintf(fp6, "apim.xbar_init(A: tile %d, mac_unit %d, x-bar %d)\n", ti, rb, xb);
        ADDR(x+i*APIM_SIMD_SIZE);
        fprintf(fp6, "apim.mvmul(x: tile %d, mac_unit %d, x-bar %d)\n", ti, rb, xb);
        ADDR(y+i*APIM_SIMD_SIZE+j);
				fprintf(fp6, "apim.store(y: tile %d, mac_unit %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE); 		
        fclose(fp6); 
      }
			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp7 = fopen("./IR/C_Func-to-IR_APIM_PRIME.trace","a");
 			  ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
        ADDR(A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
				fprintf(fp7, "apim.xbar_init(A: chip %d, reram_bank %d, x-bar %d)\n", ti, rb, xb);
        ADDR(x+i*APIM_SIMD_SIZE);
        fprintf(fp7, "apim.mvmul(x: chip %d, reram_bank %d, x-bar %d)\n", ti, rb, xb);
        ADDR(y+i*APIM_SIMD_SIZE+j);
				fprintf(fp7, "apim.store(y: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
        fclose(fp7);
      }	
      if(LANG_NUM == 1) {
        FILE* fp7 = fopen("./IR/OpenMP_Func-to-IR_APIM_PRIME.trace","a");
 			  ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
        ADDR(A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
				fprintf(fp7, "apim.xbar_init(A: chip %d, reram_bank %d, x-bar %d)\n", ti, rb, xb);
        ADDR(x+i*APIM_SIMD_SIZE);
        fprintf(fp7, "apim.mvmul(x: chip %d, reram_bank %d, x-bar %d)\n", ti, rb, xb);
        ADDR(y+i*APIM_SIMD_SIZE+j);
				fprintf(fp7, "apim.store(y: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
        fclose(fp7);
      }		
      if(LANG_NUM == 2) {
        FILE* fp7 = fopen("./IR/CUDA_Func-to-IR_APIM_PRIME.trace","a");
 			  ADDR(x+i);
				fprintf(fp7, "apim.load(x: chip %d, reram_bank %d, x-bar %d, reg: %d, %d)\n", ti, rb, xb, reg,APIM_SIMD_SIZE);
        ADDR(A+i*APIM_SIMD_SIZE*APIM_SIMD_SIZE+j*APIM_SIMD_SIZE);
				fprintf(fp7, "apim.xbar_init(A: chip %d, reram_bank %d, x-bar %d)\n", ti, rb, xb);
        ADDR(x+i*APIM_SIMD_SIZE);
        fprintf(fp7, "apim.mvmul(x: chip %d, reram_bank %d, x-bar %d)\n", ti, rb, xb);
        ADDR(y+i*APIM_SIMD_SIZE+j);
				fprintf(fp7, "apim.store(y: chip %d, reram_bank %d, x-bar %d, %d)\n", ti, rb, xb, APIM_SIMD_SIZE);
        fclose(fp7);
      }	
			#endif 
		}
	}
#endif
#ifdef DPIM
	int n2 = N/DPIM_SIMD_SIZE;
	int m2 = M/DPIM_GANG_SIZE;
	for(int i=0; i<n2; i++){
		for(int j=0; j<m2; j++){  
       #ifdef MAP_TYPE1
     if(LANG_NUM == 0) {
        FILE* fp8 = fopen("./IR/C_Func-to-IR_DPIM_Newton.trace","a"); 
  	    ADDR(x+i*DPIM_SIMD_SIZE);
  		  fprintf(fp8, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp8, "dpim.vector_comp(A: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp8, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp8);
      } 
     if(LANG_NUM == 1) {
        FILE* fp8 = fopen("./IR/OpenMP_Func-to-IR_DPIM_Newton.trace","a"); 
  	    ADDR(x+i*DPIM_SIMD_SIZE);
  		  fprintf(fp8, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp8, "dpim.vector_comp(A: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp8, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp8);
      } 
      if(LANG_NUM == 2) {
        FILE* fp8 = fopen("./IR/CUDA_Func-to-IR_DPIM_Newton.trace","a"); 
  	    ADDR(x+i*DPIM_SIMD_SIZE);
  		  fprintf(fp8, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp8, "dpim.vector_comp(A: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp8, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp8);
      } 
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp9 = fopen("./IR/C_Func-to-IR_DPIM_HBM-PIM.trace","a");
  	    ADDR(x+i*DPIM_SIMD_SIZE);
  		  fprintf(fp9, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp9, "dpim.vector_comp(A: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp9, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp9);
      } 
      if(LANG_NUM == 1) {
        FILE* fp9 = fopen("./IR/OpenMP_Func-to-IR_DPIM_HBM-PIM.trace","a");
  	    ADDR(x+i*DPIM_SIMD_SIZE);
  		  fprintf(fp9, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp9, "dpim.vector_comp(A: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp9, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp9);
      }
      if(LANG_NUM == 2) {
        FILE* fp9 = fopen("./IR/CUDA_Func-to-IR_DPIM_HBM-PIM.trace","a");
  	    ADDR(x+i*DPIM_SIMD_SIZE);
  		  fprintf(fp9, "dpim.load(x: ch %d ra %d bg %d ba %d ro %d co %d, reg: %d, %d)\n", ch, ra, bg, ba, ro, co, reg,DPIM_SIMD_SIZE); 
  		  fprintf(fp9, "dpim.vector_comp(A: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co,DPIM_SIMD_SIZE);			
				fprintf(fp9, "dpim.store(y: ch %d ra %d bg %d ba %d ro %d co %d, %d)\n", ch, ra, bg, ba, ro, co, DPIM_SIMD_SIZE);
        fclose(fp9);
      }
 			#endif    
		}
	}
#endif
#ifdef PNM
	int n3 = N/PNM_SIMD_SIZE;
	int m3 = M/PNM_SIMD_SIZE;
	for(int i=0; i<n3; i++){
		for(int j=0; j<m3; j++){		  
      #ifdef MAP_TYPE1
      if(LANG_NUM == 0) {
        FILE* fp10 = fopen("./IR/C_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE);
        fprintf(fp10, "pnm.vector(A: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(x+i*PNM_SIMD_SIZE);
        fprintf(fp10, " x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 1) {
        FILE* fp10 = fopen("./IR/OpenMP_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE);
        fprintf(fp10, "pnm.vector(A: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(x+i*PNM_SIMD_SIZE);
        fprintf(fp10, " x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
      if(LANG_NUM == 2) {
        FILE* fp10 = fopen("./IR/CUDA_Func-to-IR_PNM_RecNMP.trace","a");
        ADDR(A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE);
        fprintf(fp10, "pnm.vector(A: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(x+i*PNM_SIMD_SIZE);
        fprintf(fp10, " x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp10);
      }
 			#endif
			#ifdef MAP_TYPE2
      if(LANG_NUM == 0) {
        FILE* fp11 = fopen("./IR/C_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE);
        fprintf(fp11, "pnm.vector(A: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(x+i*PNM_SIMD_SIZE);
        fprintf(fp11, " x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 1) {
        FILE* fp11 = fopen("./IR/OpenMP_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE);
        fprintf(fp11, "pnm.vector(A: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(x+i*PNM_SIMD_SIZE);
        fprintf(fp11, " x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
      if(LANG_NUM == 2) {
        FILE* fp11 = fopen("./IR/CUDA_Func-to-IR_PNM_TeonsorDIMM.trace","a");
        ADDR(A+i*PNM_SIMD_SIZE*PNM_SIMD_SIZE+j*PNM_SIMD_SIZE);
        fprintf(fp11, "pnm.vector(A: ch %d ra %d bg %d ba %d ro %d co %d, ", ch, ra, bg, ba, ro, co);
        ADDR(x+i*PNM_SIMD_SIZE);
        fprintf(fp11, " x: ch %d ra %d bg %d ba %d ro %d co %d, %d, mul)\n", ch, ra, bg, ba, ro, co, PNM_SIMD_SIZE);
        fclose(fp11);
      }
 			#endif     
		}
	}
#endif
	for(int i=0; i<N; i++){
		VEC_MUL(x, &A[i*M], N);
		y[i]=VEC_ACC(&A[i*M], N);
	}
  MV_CYCLE+=2;
  //printf("mv mul\n");
}
void MAT_IMM_MUL(float imm, float *A, int N, int M) {
	for(int i=0; i<N; i++){
		VEC_IMM_MUL(imm, &A[i*M], M);
	}
}
void MM_ADD(float *A, float *B, int N, int M) {
	for(int i=0; i<N; i++){
		VEC_ADD(&A[i*M], &B[i*M], M);
	}
}
void MM_MUL(float* A, float *B, float *C, int N, int M, int K) {
	for(int i=0; i<N; i++){
		MV_MUL(&A[i*M], B, &C[i*K], M, K);
	}
}
#endif
