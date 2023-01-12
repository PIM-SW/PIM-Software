.RECIPEPREFIX = >
SRCDIR=src
OBJDIR=obj
IRDIR=IR
TARGET1=libopenblas.so.0.0.0
TARGET2=libopenblas.so
TARGET3=libopenblas.so.0
TEST1=testc
TEST2=testomp
TEST3=testcuda
BLAS=openblas
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

CC=clang
CFL=-shared -Wl,-soname,libopenblas.so.0
CFLAGS=-fPIC
OMP=-fopenmp

NVCC=nvcc
CUDAFL=--shared --compiler-options '-fPIC'
CUDAFLAGS=-shared -Xcompiler -fPIC

$(OBJDIR)/%.o: $(SRCDIR)/%.c
> $(CC) $(CFLAGS) -c -o $@ $<

$(TARGET1): $(OBJS)
> $(CC) $(CFL) -o $@ $(OBJS)

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TEST1) $(TEST2) $(TEST3)

$(TARGET2): | $(TARGET1)
> ln -s $(TARGET1) $@

$(TARGET3): | $(TARGET1)
> ln -s $(TARGET1) $@

$(TEST1) : $(TARGET1)
> gcc -std=c11 -o $@ $(TEST1).c -I./$(SRCDIR) -L./ -l$(BLAS)

$(TEST2) : $(TARGET1)
> gcc -std=c11 $(OMP) -o $@ $(TEST2).c -I./$(SRCDIR) -L./ -l$(BLAS)

$(TEST3) : $(TARGET1)
> $(NVCC) -o $@ $(TEST3).cu -I./$(SRCDIR) -L./ -l$(BLAS)

> cp $(TARGET1) $(TARGET2) $(TARGET3) 1_SVM-RFE/

clean:
> rm $(OBJS) $(TARGET1) $(TARGET2) $(TARGET3) $(TEST1) $(TEST2) $(TEST3) $(IRDIR)/*.trace
