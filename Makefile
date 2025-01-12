CC     = clang
CFLAGS = -Wall -Wextra -fPIC -Iinclude

LIB_NAME = lightlemur
SRCS     = backend/src/tensor.c backend/src/ops.c backend/src/binaryops.c
OBJS     = $(SRCS:.c=.o)

# Detect platform and set the target extension
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
    TARGET_EXT = so
else ifeq ($(UNAME_S), Darwin)
    TARGET_EXT = dylib
else ifeq ($(OS), Windows_NT)
    TARGET_EXT = dll
else
    $(error Unsupported platform)
endif

TARGET = lib$(LIB_NAME).$(TARGET_EXT)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
