###################################################################
#  Makefile for neural net
###################################################################

# compilers
CXX = g++
CC  = gcc
F90 = gfortran

# flags
CFLAGS   = -O2
CXXFLAGS = -O2 -std=c++11
FFLAGS   = -O2

CPPFLAGS_OMP = -Xpreprocessor -fopenmp -DOMP
CFLAGS_OMP   = -O2 -I/opt/homebrew/include
CXXFLAGS_OMP = -O2 -I/opt/homebrew/include -std=c++11
LDFLAGS_OMP  = -Wl,-rpath,/opt/homebrew/lib -L/opt/homebrew/lib -lomp

# makefile targets
all : train-mnist

train-mnist : train-mnist.cpp
	$(CXX) $(CXXFLAGS) train-mnist.cpp loadmnist.cpp layer.cpp net.cpp classifier.cpp sequential.cpp -o $@

train-mnist-omp : train-mnist.cpp
	$(CXX) $(CPPFLAGS_OMP) $(CXXFLAGS_OMP) $(LDFLAGS_OMP) train-mnist.cpp loadmnist.cpp layer.cpp net.cpp classifier.cpp sequential.cpp -o $@

temp : temp.cpp 
	$(CXX) $(CXXFLAGS) temp.cpp classifier.cpp sequential.cpp loadmnist.cpp layer.cpp net.cpp -o $@

clean :
	\rm -f *.o *.out train-mnist train-fashion train-mnist-omp temp

####### End of Makefile #######