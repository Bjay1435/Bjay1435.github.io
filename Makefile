EXECUTABLE := opencv_test

CU_FILES   := cudaDetection.cu opencvCuda.cu

CC_FILES   := opencv_test.cpp util.cpp rect.cpp cpuDetection.cpp cpuThreadDetection.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64 -fopenmp
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=

GATESINC := -I/tmp/cvinc/include -I/afs/cs/academic/class/15418-s17/public/sw/opencv/include \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/build \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/core/include \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/contrib/include \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/objdetect/include \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/imgcodecs/include/ \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/imgproc/include/ \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/highgui/include/ \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/videoio/include/ \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/imgcodecs/include/opencv2/imgcodecs \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/objdetect/include/opencv2/objdetect \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/cudaobjdetect/include/ \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/cudaimgproc/include/ \
-I/afs/cs/academic/class/15418-s17/public/sw/opencv/modules/cudawarping/include/ \



NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -g -G -use_fast_math
LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))

# opencv linking `pkg-config --libs opencv` -L/afs/cs.cmu.edu.acadmeic/class/15418-s17/public/sw/opencv/build/lib
LDFLAGS	:= -pthread -L/usr/local/cuda/lib64/ -L/afs/cs/academic/class/15418-s17/public/sw/opencv/build/lib `pkg-config --libs opencv` 

NVCC=nvcc

OBJS=$(OBJDIR)/util.o $(OBJDIR)/rect.o $(OBJDIR)/opencvCuda.o $(OBJDIR)/cudaDetection.o $(OBJDIR)/cpuThreadDetection.o $(OBJDIR)/cpuDetection.o $(OBJDIR)/opencv_test.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs: 
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@ $(GATESINC)

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@ $(GATESINC)