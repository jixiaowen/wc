CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c99
TARGET ?= wc

all: $(TARGET)

$(TARGET): wc.c
	$(CC) $(CFLAGS) -o $(TARGET) wc.c

clean:
	rm -f $(TARGET)
