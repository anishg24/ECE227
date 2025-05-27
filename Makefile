
all: fast_ic

fast_ic: src/independent_cascade.cpp
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(shell python3 -m pybind11 --includes) src/independent_cascade.cpp -o src/fast_ic$(shell python3-config --extension-suffix)

clean:
	rm -f fast_ic*.so
