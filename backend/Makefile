CC     = clang
CFLAGS = -Wall -Wextra -fPIC -Iinclude

LIB_NAME = lightlemur
TARGET   = lib$(LIB_NAME).dylib #.so for linux

SRCS = src/tensor.c src/ops.c src/binaryops.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
