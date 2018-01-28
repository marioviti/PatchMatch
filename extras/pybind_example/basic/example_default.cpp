#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int i = 1, int j = 2) {
  return i+j;
}

PYBIND11_MODULE(example_default, m) {
  m.doc() = "pybind11 example_default plugin";
  // regular notation
  m.def("add", &add, "A function which adds two numbers",
        py::arg("i") = 1, py::arg("j") = 2);
}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example_default.cpp -o example_default`python3-config --extension-suffix`
*/
