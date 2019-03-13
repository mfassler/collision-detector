######
######   What are we building?
######

TARGET = alertIfTooClose


# Objects that must be built in order to link

OBJECTS = main.o
OBJECTS += Darknet.o

######
######   Binaries and flags
######

CPPFLAGS = -std=c++11
CPPFLAGS += -O3
#CPPFLAGS += -g

LD = g++

#LDFLAGS = -L/usr/local/lib/
LDFLAGS = -pthread
LDLIBS += -lrealsense2
LDLIBS += -ldarknet
LDLIBS += $(shell pkg-config --libs opencv)


# Default target:
.PHONY: all
all: $(TARGET)


$(TARGET): $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o $@ $(LDLIBS)


.PHONY: clean
clean:
	rm -f $(OBJECTS)
	rm -f $(TARGET)


