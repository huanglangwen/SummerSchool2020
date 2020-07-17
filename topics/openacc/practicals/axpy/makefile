.PHONY: all clean

#CRAY ?= 0
#PGI  ?= 1
#GCC  ?= 0
VERBOSE ?= 0
override CPPFLAGS += -I../common

CXX = pgcc
FC  = pgf90
override CUDA_PATH = C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v11.0
CXXFLAGS = -DPGI -O3 -acc -Minfo=accel
FCFLAGS  = -O3 -acc -Minfo=accel
LDFLAGS += -L"$(CUDA_PATH)/lib/x64" -L:cudart.lib

OPENACC_BENCHMARKS= axpy.openacc axpy.openacc.fort

BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

util.obj: ../common/util.f03
	$(FC) $(CPPFLAGS) -c $(FCFLAGS) $<

axpy.openacc: axpy_openacc.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

axpy.openacc.fort: axpy_openacc.f90 util.obj
	$(FC) $(CPPFLAGS) $(FCFLAGS) $< util.obj -o $@ $(LDFLAGS)

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.obj *.mod
