.RECIPEPREFIX = >
SRCDIR=../src
OBJDIR=../obj
LIBDIR=../lib

TARGET=test

INCLUDE_PATH=../include
LIB_PATH=../lib
LIB=lpimblas

CC=clang
CFLG=-std=c11 -g -Wall -fopenmp

all: $(TARGET)

$(TARGET): 
> $(CC) $(CFLG) -o ../$@ test.c -I$(INCLUDE_PATH) -L$(LIB_PATH) -lpimblas

clean:
> rm ../$(TARGET)
