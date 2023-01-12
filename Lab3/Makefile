MPICXX=mpiicpc

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS=-O0 -g -std=c++11 -no-multibyte-chars
else
	CXXFLAGS=-O3 -g -std=c++11 -no-multibyte-chars
endif

EXECUTABLE =  summa summa_nb summa_check summa_nb_check
SRC=summa.cpp
all: $(EXECUTABLE)

team: 
	@echo "Abhishek Raj\nUtkarsh Prakash"

summa: summa.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^

summa_nb: summa.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^

summa_check: summa.cpp
	$(MPICXX) -DCHECK_NUMERICS $(CXXFLAGS) -o $@ $^

summa_nb_check: summa.cpp
	$(MPICXX) -DCHECK_NUMERICS $(CXXFLAGS) -o $@ $^

clean:
	rm -f *.o *~ core $(EXECUTABLE)
