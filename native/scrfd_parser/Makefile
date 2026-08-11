TARGET := libnvdsinfer_custom_scrfd.so
CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
CXXFLAGS := -O3 -Wall -Wextra -std=c++17 -shared -fPIC
CPPFLAGS := -I/opt/nvidia/deepstream/deepstream/sources/includes -I$(CUDA_HOME)/include

.PHONY: all clean

all: $(TARGET)

$(TARGET): nvdsparsebbox_SCRFD.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
