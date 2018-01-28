#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void put_zeros(py::array_t<double> a) {
  auto r = a.mutable_unchecked<2>();
  auto buf = a.request();
  for (size_t idx = 0; idx < buf.shape[0]; idx++)
    for (size_t idy = 0; idy < buf.shape [1]; idy++)
      r(idx,idy) = 0.f;
}

py::array_t<double> convolution(py::array_t<double> image, py::array_t<double> kernel) {
    auto bufImage = image.request(), bufKernel = kernel.request();

    if (bufImage.ndim != 2 || bufKernel.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    auto i = image.mutable_unchecked<2>();
    auto k = kernel.mutable_unchecked<2>();

    auto result = py::array_t<double>(bufImage.shape);
    auto bufResult = result.request();
    auto r = result.mutable_unchecked<2>();

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

bool is_clamped(int h, int w, int i0, int i1, int j0, int j1) {
  bool p1 =   i0 < h and i0 > 0.f;
  p1 = p1 and i1 < h and i1 > 0.f;
  p1 = p1 and j0 < w and j0 > 0.f;
  p1 = p1 and j1 < w and j1 > 0.f;
  return p1;
}

float ssd(py::array_t<double> A, py::array_t<double> B,
          int a_i, int a_j,
          int b_i, int b_j, int w) {
  auto bufA = A.request();
  auto pA = A.mutable_unchecked<3>();
  auto bufB = B.request();
  auto pB = B.mutable_unchecked<3>();

  int channels = bufA.shape[2];
  float sq_su = 0.0;
  for (size_t offx = 0; offx < w; offx++)
    for (size_t offy = 0; offy < w; offy++)
      for (size_t c = 0; c < channels; c++){
        sq_su += std::pow(pA(a_i+offx,a_j+offy,c) - pB(b_i+offx,b_j+offy,c),2);
      }
  return sq_su;
}

void propagation(int i, int j, py::array_t<double> A, py::array_t<double> B,
                  py::array_t<double> nnf, int patch_size, int prop_step) {
    auto bufnnf = nnf.request();
    int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
    auto pnnf = nnf.mutable_unchecked<3>();

    auto bufB = B.request();
    int B_h = bufB.shape[0], B_w = bufB.shape[1];
    auto pB = B.mutable_unchecked<3>();

    // curr min offset
    float min_i = pnnf(i,j,0), min_j = pnnf(i,j,1), min_dist = pnnf(i,j,2);
    bool update = false;

    // shifted offset on h
    float fi = pnnf(i+prop_step,j,0), fj = pnnf(i+prop_step,j,1);
    float curr_dist = pnnf(i+prop_step,j,2);

    if (curr_dist < min_dist) {
      if (is_clamped(B_h,B_w,
                    fi-prop_step,fi-prop_step+patch_size,
                    fj,fj+patch_size)){
        min_i = fi-prop_step;
        min_j = fj;
      } else {
        min_i = fi;
        min_j = fj;
      }
      min_dist = curr_dist;
      update = true;
    }

    // shiftedn offset on w
    fi = pnnf(i,j+prop_step,0), fj = pnnf(i,j+prop_step,1);
    curr_dist = pnnf(i,j+prop_step,2);

    if (curr_dist < min_dist) {
      if (is_clamped(B_h,B_w,
                    fi,fi+patch_size,
                    fj-prop_step ,fj-prop_step+patch_size)){
        min_i = fi;
        min_j = fj-prop_step;
      } else {
        min_i = fi;
        min_j = fj;
      }
      min_dist = curr_dist;
      update = true;
    }

    pnnf(i,j,0) = min_i;
    pnnf(i,j,1) = min_j;
    if (update) {
      int a_i = i, a_j = j;
      int b_i = int(min_i), b_j = int(min_j);
      pnnf(i,j,1) = ssd(A,B,a_i,a_j,b_i,b_j,patch_size);
    }
}

void iteration(py::array_t<double> A, py::array_t<double> B, py::array_t<double> nnf,
                int patch_size, int even, int step ) {
    auto bufnnf = nnf.request();
    int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
    if (even == 1){
      //even iteration propagation: scan order (l-r,t-b)
      int lstep = -step;
      int inc_left_h = step, exc_right_h = nnf_h, d = 1;
      int inc_left_w = step, exc_right_w = nnf_w;
      for (size_t i = inc_left_h; i < exc_right_h; i+=d)
        for (size_t j = inc_left_w; i < exc_right_w; j+=d)
          propagation(i,j,A,B,nnf,patch_size,lstep);
          //random_search(i,j,A,B,nnf,patch_size);
    }
    else {
      //odd iteration\npropagation: inverse scan order (t-b,r-l)
      int lstep = step;
      int inc_left_h = (nnf_h-1)-step, exc_right_h = -1, d = -1;
      int inc_left_w =(nnf_w-1)-step, exc_right_w = -1;
      for (size_t i = inc_left_h; i > exc_right_h; i+=d)
        for (size_t j = inc_left_w; i > exc_right_w; j+=d)
          propagation(i,j,A,B,nnf,patch_size,lstep);
          //random_search(i,j,A,B,nnf,patch_size);
    }
}

PYBIND11_MODULE(patchmatch, m) {
    m.def("convolution", &convolution, "convolve two NumPy arrays");
}
/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` patchmatch.cpp -o patchmatch`python3-config --extension-suffix`
*/
