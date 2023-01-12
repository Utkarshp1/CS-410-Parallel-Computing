NVCXX=nvcc

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS=-O0 -g -std=c++11
else
	CXXFLAGS=-O3 -g -std=c++11
endif

EXECUTABLE =  ver0 ver1 ver0_check ver1_check
SRC=ver0.cu
all: $(EXECUTABLE)

team: 
	@echo "Abhishek Raj\nUtkarsh Prakash"

exp0: ver0
	./$^ 1024
	./$^ 2048
	./$^ 4096
	./$^ 8192

exp1: ver1
	./$^ 1024
	./$^ 2048
	./$^ 4096
	./$^ 8192

ver0: ver0.cu
	$(NVCXX) $(CXXFLAGS) -o $@ $^

ver1: ver1.cu
	$(NVCXX) $(CXXFLAGS) -o $@ $^

ver0_check: ver0.cu
	$(NVCXX) -DVALIDATE $(CXXFLAGS) -o $@ $^

ver1_check: ver1.cu
	$(NVCXX) -DVALIDATE $(CXXFLAGS) -o $@ $^

clean:
	rm -f *.o *~ core $(EXECUTABLE)
