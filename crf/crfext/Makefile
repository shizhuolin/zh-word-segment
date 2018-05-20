
CXX = g++
CXXFLAGS += -std=c++0x -Wall -fexceptions -fPIC -fopenmp -O3 -march=native
CXXFLAGS += -I/usr/include/python3.4m/
CXXFLAGS += -lgomp -shared

all:
	$(CXX) $(CXXFLAGS) main.cpp -o ../crfext.so

clean:
	$(RM) ../crfext.so

Debug:all

Release:all

cleanRelease:clean
