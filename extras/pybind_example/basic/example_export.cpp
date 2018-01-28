#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(example_export, m) {
  m.doc() = "pybind11 example_export plugin";
  m.attr("the_answer") = 42;
  py::object world = py::cast("World");
  m.attr("what") = world;
}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example_export.cpp -o example_export`python3-config --extension-suffix`
*/
