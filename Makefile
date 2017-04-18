EXECUTABLE := opencv_test

CC_FILES   := opencv_test.cpp util.cpp rect.cpp cpuDetection.cpp cpuThreadDetection

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

# opencv linking
LDFLAGS	:= `pkg-config --libs opencv` -pthread

OBJS=$(OBJDIR)/util.o $(OBJDIR)/rect.o $(OBJDIR)/cpuThreadDetection.o $(OBJDIR)/cpuDetection.o $(OBJDIR)/opencv_test.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs: 
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@
