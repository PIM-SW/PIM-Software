.RECIPEPREFIX = >
SRCDIR=src
OBJDIR=obj
OCLDIR=usr/local/cuda-11.0/targets/x86_64-linux/include/
TARGET1=libopenblas.so
TEST1=testc
TEST2=testomp
TEST3=testcuda
TEST4=testocl
BLAS=openblas
OCL=OpenCL
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

CC=clang
CFL=-shared -Wl,-soname,libopenblas.so
CFLAGS=-fPIC
OMP=-fopenmp

$(OBJDIR)/%.o: $(SRCDIR)/%.c
> $(CC) $(CFLAGS) -c -o $@ $<

$(TARGET1): $(OBJS)
> $(CC) $(CFL) -o $@ $(OBJS)

all: $(TARGET1) $(TEST1) $(TEST2) $(TEST3) $(TEST4)

$(TEST1) : $(TARGET1)
> gcc -std=c11 -o $@ $(TEST1).c -I./$(SRCDIR) -L./ -l$(BLAS)

$(TEST2) : $(TARGET1)
> gcc -std=c11 $(OMP) -o $@ $(TEST2).c -I./$(SRCDIR) -L./ -l$(BLAS)

$(TEST3) : $(TARGET1)
> nvcc -o $@ $(TEST3).cu -I./$(SRCDIR) -L./ -l$(BLAS)

$(TEST4) : $(TARGET1)
> gcc -std=c11 -o $@ $(TEST4).c -I./$(SRCDIR) -L./ -l$(BLAS) -l $(OCL) -I/$(OCLDIR)

clean:
> rm $(OBJS) $(TARGET1) $(TEST1) $(TEST2) $(TEST3) $(TEST4)
