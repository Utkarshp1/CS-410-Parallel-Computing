CXX=g++

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS=-O0 -g -std=c++11 
else
	CXXFLAGS=-O3 -g -std=c++11
endif

EXECUTABLE=ver0 ver1_omp ver1_pt ver2_omp ver2_pt ver3_omp ver3_pt a.out

team:
	@echo "Abhishek Raj\nUtkarsh Prakash"

exp1: ver0
	./$< 720 1
	./$< 1440 1
	./$< 2160 1

exp2: ver1_omp ver1_pt
	chmod +x exp2.sh
	./exp2.sh $^

exp3: ver1_omp
	chmod +x exp3.sh
	./exp3.sh $^

ver0: ver0.cpp
	$(CXX) -o $@ ver0.cpp $(CXXFLAGS)

ver1_omp: ver1_omp.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) -fopenmp

ver1_pt: ver1_pt.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) -pthread

ver2_omp: ver2_omp.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) -fopenmp

ver2_pt: ver2_pt.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) -pthread

ver3_omp: ver3_omp.cpp
	$(CXX) -I/home/iit/cs410software/openblas/include $< -L/home/iit/cs410software/openblas/lib -lopenblas -o $@ $(CXXFLAGS) -fopenmp

ver3_pt: ver3_pt.cpp
	$(CXX) -I/home/iit/cs410software/openblas/include $< -L/home/iit/cs410software/openblas/lib -lopenblas -o $@ $(CXXFLAGS) -pthread

clean:
	rm -f *.o *~ core $(EXECUTABLE)