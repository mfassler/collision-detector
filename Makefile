## Build Options:
NETWORK_DISPLAY = 0
LOCAL_DISPLAY = 1

######
######   What are we building?
######

TARGET = alertIfTooClose


# Objects that must be built in order to link

OBJECTS = main.o
OBJECTS += CDNeuralNet.o

######
######   Binaries and flags
######

CPPFLAGS = -std=c++11
ifeq ($(LOCAL_DISPLAY), 1)
CPPFLAGS += -DUSE_LOCAL_DISPLAY
endif

ifeq ($(NETWORK_DISPLAY), 1)
CPPFLAGS += -DUSE_NETWORK_DISPLAY -DUSE_SCANLINE
OBJECTS += UdpSender.o
endif

CPPFLAGS += -O3
CPPFLAGS += -I/usr/include/opencv4
#CPPFLAGS += -g

LD = g++

#LDFLAGS = -L/usr/local/lib/
LDFLAGS = -pthread
LDLIBS += -lrealsense2
LDLIBS += $(shell pkg-config --libs opencv4)


# Default target:
.PHONY: all
all: $(TARGET)


$(TARGET): $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o $@ $(LDLIBS)


.PHONY: clean
clean:
	rm -f $(OBJECTS)
	rm -f $(TARGET)


