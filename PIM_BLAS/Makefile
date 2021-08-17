.RECIPEPREFIX = >
SRCDIR=src
OBJDIR=obj
TARGET=micro_bench
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

CC=clang
CFLAGS=-std=c11 -g -Wall

$(TARGET): $(OBJS)
> $(CC) $(CFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
> $(CC) $(CFLAGS) -c -o $@ $<

clean:
> rm $(OBJS) $(TARGET)
