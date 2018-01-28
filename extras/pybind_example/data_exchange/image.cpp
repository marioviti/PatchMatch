#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void put_zeros(py::array_t<double> a) {
  auto r = a.mutable_unchecked<2>();
  auto bufa = a.request();
  for (size_t idx = 0; idx < bufa.shape[0]; idx++)
    for (size_t idy = 0; idy < bufa.shape[1]; idy++)
      r(idx,idy) = 0.f;
}

py::array_t<double> convolution(py::array_t<double> image, py::array_t<double> kernel) {
    auto bufImage = image.request(), bufKernel = kernel.request();

    if (bufImage.ndim != 2 || bufKernel.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    auto result = py::array_t<double>(bufImage.shape);
    auto bufResult = result.request();
    auto r = result.mutable_unchecked<2>();
    auto i = image.mutable_unchecked<2>();
    auto k = kernel.mutable_unchecked<2>();

    put_zeros(result);

    // convolution
    int h_w = int(std::floor((bufKernel.shape[0]/2)));
    for (size_t idx = h_w; idx < bufImage.shape[0]-h_w; idx++)
      for (size_t idy = h_w; idy < bufImage.shape[1]-h_w; idy++)
        for (size_t id2x = 0; id2x < bufKernel.shape[0]; id2x++)
          for (size_t id2y = 0; id2y < bufKernel.shape[1]; id2y++)
            r(idx,idy) += i(idx-h_w+id2x,idy-h_w+id2y) * k(id2x,id2y);

    return result;
}

PYBIND11_MODULE(image, m) {
    m.def("convolution", &convolution, "convolve two NumPy arrays");
}
/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` image.cpp -o image`python3-config --extension-suffix`
*/
