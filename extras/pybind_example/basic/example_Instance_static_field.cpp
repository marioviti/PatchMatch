#include <pybind11/pybind11.h>
namespace py = pybind11;

struct Pet {
  Pet(const std::string & name) : name(name) {}
  void setName(const std::string & name_) {name = name_;}
  const std::string &getName() const {return name;}

  std::string name;
};

PYBIND11_MODULE(example_Instance_static_field, m) {
  py::class_<Pet>(m,"Pet")
    .def(py::init<const std::string &>())
    .def_readwrite("name", &Pet::name)
    .def("setName", &Pet::setName)
    .def("getName", &Pet::getName)
    .def("__repr__",
        [](const Pet &a){
          return "<Pet named '" + a.name + "'>";
        }
      );
}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example_Instance_static_field.cpp -o example_Instance_static_field`python3-config --extension-suffix`
*/
