CXX_CC := g++
CXX_FLAGS := -std=c++11 -g -Xcompiler -fopenmp -I include

CUDA_CC := nvcc
CUDA_FLAGS := -O3 -use_fast_math -extra-device-vectorization
CUDA_ARCH_FRONTAL := -gencode arch=compute_75,code=sm_75
CUDA_ARCH_GPUNODE := -gencode arch=compute_60,code=sm_60

SRCDIR := src
INCDIR := include
BINDIR := target
DEPDIR := $(BINDIR)/deps
BINFRONTAL := $(BINDIR)/frontal-cuda-dgemm
BINGPUNODE := $(BINDIR)/gpunode-cuda-dgemm
RESDIR := results

.PHONY: all clean

all: $(BINFRONTAL) $(BINGPUNODE)

$(BINFRONTAL): $(SRCDIR)/kernels.cu $(SRCDIR)/drivers.cu $(SRCDIR)/utils.cpp $(SRCDIR)/main.cpp
	@mkdir -p $(BINDIR)
	$(CUDA_CC) $(CUDA_ARCH_FRONTAL) $(CXX_FLAGS) $(CUDA_FLAGS) $^ -o $@ -lgomp

$(BINGPUNODE): $(SRCDIR)/kernels.cu $(SRCDIR)/drivers.cu $(SRCDIR)/utils.cpp $(SRCDIR)/main.cpp
	@mkdir -p $(BINDIR)
	$(CUDA_CC) $(CUDA_ARCH_GPUNODE) $(CXX_FLAGS) $(CUDA_FLAGS) $^ -o $@ -lgomp

clean:
	@rm -rf $(BINDIR) $(RESDIR)/*
