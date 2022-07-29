all: bin/main

bin/main: src/main.cu src/main.h
	mkdir -pv bin
	nvcc src/main.cu -arch=sm_60 -o bin/main -I/usr/include/SDL2 -D_REENTRANT -lSDL2

.PHONY:
clean:
	rm -rvf bin

.PHONY:
lint:
	clang-format -i src/*

