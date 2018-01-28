#include <pybind11/pybind11.h>
namespace py = pybind11;

struct Pet {
public:
  Pet(const std::string & name) : name(name) {}
  void setName(const std::string & name_) {name = name_;}
  const std::string &getName() const {return name;}
private:
  std::string name;
};

PYBIND11_MODULE(example_Instance_static_field_private, m) {
  py::class_<Pet>(m,"Pet")
    .def(py::init<const std::string &>())
    .def_property("name", &Pet::getName, &Pet::setName)
    .def("setName", &Pet::setName)
    .def("getName", &Pet::getName);
}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example_Instance_static_field_private.cpp -o example_Instance_static_field_private`python3-config --extension-suffix`
*/
