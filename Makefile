EXECUTABLE := opencv_test

CU_FILES   := cudaDetection.cu

CC_FILES   := opencv_test.cpp util.cpp rect.cpp cpuDetection.cpp cpuThreadDetection.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

GATESINC := -I/tmp/cvinc/include

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35 -g -G
LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

# opencv linking
LDFLAGS	:= `pkg-config --libs opencv` -pthread -L/usr/local/cuda/lib64/

NVCC=nvcc

OBJS=$(OBJDIR)/util.o $(OBJDIR)/rect.o $(OBJDIR)/cudaDetection.o $(OBJDIR)/cpuThreadDetection.o $(OBJDIR)/cpuDetection.o $(OBJDIR)/opencv_test.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs: 
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@ $(GATESINC)

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@  $(GATESINC)