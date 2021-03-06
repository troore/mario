BIN := cuda_md5

# flags
CUDA_INSTALL_HOME := /usr/local/cuda
CUDA_SDK_HOME := /home/weixuechao/Projects/cuda/NVIDIA_GPU_Computing_SDK

CUTIL_LIB_TYPE := $(if $(shell uname -m | grep 64),x86_64,i386)
CUDA_INSTALL_LIB := $(if $(shell uname -m | grep 64),lib64,lib)

COMMONFLAGS := -I.
NVCCFLAGS := -g -G -arch=sm_13 --ptxas-options=-v --use_fast_math -I$(CUDA_INSTALL_HOME)/include $(COMMONFLAGS)
CXXFLAGS := -O3 -g -Wall $(COMMONFLAGS) -I$(CUDA_INSTALL_HOME)/include

LIB	:= -lcutil_$(CUTIL_LIB_TYPE) -lcudart
LIBPATH	:= -L$(CUDA_SDK_HOME)/C/lib -L$(CUDA_SDK_HOME)/C/common/lib -L$(CUDA_SDK_HOME)/shared/lib -L$(CUDA_INSTALL_HOME)/$(CUDA_INSTALL_LIB)
LINKFLAGS := -Wl,-rpath=$(CUDA_INSTALL_HOME)/$(CUDA_INSTALL_LIB) -lm $(LIB)  $(LIBPATH)
# compilers
NVCC              := nvcc

# files
CPP_SOURCES       := cuda_md5.cpp cuda_md5_cpu.cpp
CU_SOURCES        := cuda_md5_gpu.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu cuda_md5.h
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o: %.cpp cuda_md5.h
	$(CXX) $(CXXFLAGS) -o $@  -c $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LINKFLAGS)

clean:
	rm -f $(BIN) *.o *.cu_o
