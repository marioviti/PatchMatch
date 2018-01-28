#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int i, int j) {
  return i+j;
}

PYBIND11_MODULE(example_kw, m) {
  m.doc() = "pybind11 example plugin";
  m.def("add", &add, "A function which adds two numbers",
        py::arg("i"), py::arg("j"));
}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example_kw.cpp -o example_kw`python3-config --extension-suffix`
*/
