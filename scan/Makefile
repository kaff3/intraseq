CXX        = nvcc -O3

#-arch=sm_60

SRC = main.cu
HELPERS = scan.cuh
EXECUTABLE = main

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

