SRCS=patchmatch.cpp

patchmatch`python3-config --extension-suffix`: $(SRCS)
	c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` patchmatch.cpp -o patchmatch`python3-config --extension-suffix`
