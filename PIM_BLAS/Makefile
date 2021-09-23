.RECIPEPREFIX = >
SRCDIR=src
OBJDIR=obj
TARGET1=libopenblas.so.0.0.0
TARGET2=libopenblas.so
TARGET3=libopenblas.so.0
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

CC=clang
CFL=-shared -Wl,-soname,libopenblas.so.0
CFLAGS=-fPIC

$(OBJDIR)/%.o: $(SRCDIR)/%.c
> $(CC) $(CFLAGS) -c -o $@ $<

$(TARGET1): $(OBJS)
> $(CC) $(CFL) -o $@ $(OBJS)

all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET2): | $(TARGET1)
> ln -s $(TARGET1) $@

$(TARGET3): | $(TARGET1)
> ln -s $(TARGET1) $@


clean:
> rm $(OBJS) $(TARGET1) $(TARGET2) $(TARGET3)
