EXECUTABLE := opencv_test


CC_FILES   := opencv_test.cpp util.cpp cpuDetection.cpp

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
LDFLAGS	:= `pkg-config --libs opencv`

OBJS=$(OBJDIR)/opencv_test.o $(OBJDIR)/util.o $(OBJDIR)/cpuDetection.o


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
