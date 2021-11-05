.RECIPEPREFIX = >
CC=clang
CFL=-shared -Wl,-soname,libpimblas.so
CFLAGS=-fPIC

SRCDIR=../src
OBJDIR=../obj
LIBDIR=../lib
INCLUDE_PATH=../include

SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

all: $(LIBDIR)/libpimblas.so

LIB=libpimblas.so.0.0.0

$(LIBDIR)/$(LIB): $(OBJS)
> $(CC) $(CFL) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
> $(CC) $(CFLAGS) -c -I$(INCLUDE_PATH) -o $@ $<

$(LIBDIR)/libpimblas.so: $(LIBDIR)/libpimblas.so.0.0.0
> ln -s $(LIBDIR)/libpimblas.so.0.0.0 $@

clean:
> rm $(OBJS) $(LIBDIR)/*