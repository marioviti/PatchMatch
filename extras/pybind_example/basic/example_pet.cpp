#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

struct Pet {
  Pet(const std::string & name) : name(name) {}
  void setName(const std::string & name_) {name = name_;}
  const std::string &getName() const {return name;}

  std::string name;
};

PYBIND11_MODULE(example_pet, m) {
  py::class_<Pet>(m,"Pet")
    .def(py::init<const std::string &>())
    .def("setName", &Pet::setName)
    .def("getName", &Pet::getName);

}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example_pet.cpp -o example_pet`python3-config --extension-suffix`
*/
