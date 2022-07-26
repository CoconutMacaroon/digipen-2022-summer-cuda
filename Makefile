all: bin/main

bin/main: src/main.cu src/main.h
	mkdir -pv bin
	nvcc src/main.cu -o bin/main

.PHONY:
clean:
	rm -rvf bin
