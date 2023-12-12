from ctypes import cdll, c_int, c_float
from enum import Enum

# Load shared library
libopenblas = cdll.LoadLibrary('./libopenblas.so')

# Size of the arrays
N = 36

# Initialize arrays as in testomp.c
x = (c_float * N)(*[35 - j for j in range(N)])
y = (c_float * N)(*[i for i in range(N)])
alpha = c_float(2.0)
beta = c_float(7.0)

# Call the cblas_sscal function
libopenblas.cblas_sscal(c_int(18), alpha, y, c_int(2))
print("Python result of cblas_sscal")
#for i in range(N):
    #print(y[i])

# TODO: Call other functions similarly and print results

# For functions that are not directly accessible via ctypes (e.g. OpenMP-related features), you might need to wrap them in C functions in your shared library and then call those wrapper functions from Python.

# cblas_sdsdot
result_sdsdot = libopenblas.cblas_sdsdot(c_int(18), alpha, x, c_int(2), y, c_int(2))
print("Python result of cblas_sdsdot")
#for i in range(N):
    #print(y[i])
print()

# cblas_sdot
result_sdot = libopenblas.cblas_sdot(c_int(18), x, c_int(2), y, c_int(2))
print("Python result of cblas_sdot")
#for i in range(N):
    #print(y[i])
print()

# cblas_saxpy
libopenblas.cblas_saxpy(c_int(18), alpha, x, c_int(2), y, c_int(2))
print("Python result of cblas_saxpy")
#for i in range(N):
    #print(y[i])
print()

# Initialize A, B, C, X, Y arrays
A = (c_float * N)(*[7 * (35 - j) for j in range(N)])
B = (c_float * N)(*[3 * i for i in range(N)])
C = (c_float * N)(*[3 * i for i in range(N)])
X = (c_float * 18)(*[9 * i for i in range(18)])
Y = (c_float * 12)(*[4 * i for i in range(12)])

class CBLAS_ORDER(Enum):
  CblasRowMajor=101
  CblasColMajor=102

class CBLAS_TRANSPOSE(Enum):
  CblasNoTrans=111
  CblasTrans=112
  CblasConjTrans=113
  CblasConjNoTrans=114

print("testpython Successfully Executed !!!")

# cblas_sgemv
# libopenblas.cblas_sgemv(CBLAS_ORDER.CblasColMajor.value, CBLAS_TRANSPOSE.CblasNoTrans.value, c_int(6), c_int(6), alpha, A, c_int(6), X, c_int(3), beta, Y, c_int(2))
# print("result of cblas_sgemv")
# for i in range(12):
#   print(Y[i])
# print()

# cblas_sgemm
# libopenblas.cblas_sgemm(CBLAS_ORDER.CblasColMajor.value, CBLAS_TRANSPOSE.CblasNoTrans.value, CBLAS_TRANSPOSE.CblasNoTrans.value, c_int(6), c_int(6), c_int(6), alpha, A, c_int(6), B, c_int(6), beta, C, c_int(6))
# print("result of cblas_sgemm")
# for i in range(N):
#   print(C[i])
# print()

