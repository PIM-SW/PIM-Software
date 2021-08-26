.RECIPEPREFIX = >
SRCDIR=src
OBJDIR=obj
TARGET=escal_cblas
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

CC=clang
CFLAGS=-std=c11 -g -Wall

all: $(TARGET)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
> $(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
> $(CC) $(CFLAGS) -o $@ $(OBJS)


clean:
> rm $(OBJS) $(TARGET)
