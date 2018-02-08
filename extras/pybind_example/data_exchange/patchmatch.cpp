#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>

namespace py = pybind11;

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
  int Bh = bufB.shape[0];
  int Bw = bufB.shape[1];
  int Ah = bufA.shape[0];
  int Aw = bufA.shape[1];
  int channels = bufA.shape[2];

  double sq_su = 0.0;
  for (size_t offx = 0; offx < w; offx++)
    for (size_t offy = 0; offy < w; offy++)
      for (size_t c = 0; c < channels; c++){
        sq_su += std::pow(pA(a_i+offx,a_j+offy,c) - pB(b_i+offx,b_j+offy,c),2);
      }
  return sq_su;
}


void initialization( py::array_t<double> A,
                     py::array_t<double> B,
                     py::array_t<double> nnf,
                     int patch_size) {
  auto bufnnf = nnf.request();
  auto pnnf = nnf.mutable_unchecked<3>();
  int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
  int nnf_i,nnf_j;
  double ri,rj;
  for (size_t i=0; i<nnf_h; i++)
    for (size_t j=0; j<nnf_w; j++) {
      ri = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
      rj = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
      nnf_i = static_cast <int> (floor(ri*nnf_h));
      nnf_j = static_cast <int> (floor(rj*nnf_w));
      pnnf(i,j,0) = static_cast <double> (nnf_i);
      pnnf(i,j,1) = static_cast <double> (nnf_j);
      pnnf(i,j,2) = ssd(A,B,i,j,nnf_i,nnf_j,patch_size);
    }
}


void propagation(int i, int j, py::array_t<double> A, py::array_t<double> B,
                  py::array_t<double> nnf, int patch_size, int prop_step) {
    auto bufnnf = nnf.request();
    int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
    auto pnnf = nnf.mutable_unchecked<3>();

    auto bufB = B.request();
    int B_h = bufB.shape[0], B_w = bufB.shape[1];

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
    fi = pnnf(i,j+prop_step,0);
    fj = pnnf(i,j+prop_step,1);
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
      int b_i = static_cast <int> (min_i);
      int b_j = static_cast <int> (min_j);
      pnnf(i,j,2) = ssd(A,B,a_i,a_j,b_i,b_j,patch_size);
    }
}

void random_search(int i, int j,
                    py::array_t<double> A, py::array_t<double> B,
                    py::array_t<double> nnf, int patch_size) {
  auto bufnnf = nnf.request();
  int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
  auto pnnf = nnf.mutable_unchecked<3>();
  auto bufB = B.request();
  int B_h = bufB.shape[0], B_w = bufB.shape[1];
  float ri_t, rj_t, new_distance;
  int w, u_i_t, u_j_t;

  w = nnf_h;
  if (nnf_h < nnf_w) w = nnf_w;
  float v_i_0 = pnnf(i,j,0), v_j_0 = pnnf(i,j,1), curr_min_dist = pnnf(i,j,2);

  while(w > 1) {
    ri_t = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))-1.0;
    rj_t = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))-1.0;

    u_i_t = static_cast <int> (v_i_0 + floor(w*ri_t));
    u_j_t = static_cast <int> (v_j_0 + floor(w*rj_t));

    if (is_clamped(B_h,B_w, u_i_t,u_i_t+patch_size, u_j_t,u_j_t+patch_size)) {
      new_distance = ssd(A,B,i,j,u_i_t,u_j_t,patch_size);
      if (new_distance < curr_min_dist) {
        curr_min_dist = new_distance;
        pnnf(i,j,0) = u_i_t;
        pnnf(i,j,1) = u_j_t;
        pnnf(i,j,2) = curr_min_dist;
      }
    }
    w = w/2;
  }
}

void iteration(py::array_t<double> A, py::array_t<double> B,
               py::array_t<double> nnf,
               int patch_size, bool even, int step ) {
    //initialization(A,B,nnf,patch_size);
    auto bufnnf = nnf.request();
    int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
    if (even){
      //even iteration propagation: scan order (l-r,t-b)
      int inc_left_h = step, exc_right_h = nnf_h;
      int inc_left_w = step, exc_right_w = nnf_w;
      for (int i = inc_left_h; i < exc_right_h; i+=1) {
        for (int j = inc_left_w; j < exc_right_w; j+=1) {
          propagation(i,j,A,B,nnf,patch_size,-step);
          random_search(i,j,A,B,nnf,patch_size);
        }
      }
    }
    else {
      //odd iteration\npropagation: inverse scan order (t-b,r-l)
      int inc_left_h = (nnf_h-1)-step, exc_right_h = -1;
      int inc_left_w =(nnf_w-1)-step, exc_right_w = -1;
      for (int i = inc_left_h; i > exc_right_h; i-=1) {
        for (int j = inc_left_w; j > exc_right_w; j-=1) {
          propagation(i,j,A,B,nnf,patch_size,step);
          random_search(i,j,A,B,nnf,patch_size);
        }
      }
    }
}

void put_val(py::array_t<double> a, double val) {
  auto pa = a.mutable_unchecked<3>();
  auto bufa = a.request();
  for (size_t idx = 0; idx < bufa.shape[0]; idx++)
    for (size_t idy = 0; idy < bufa.shape[1]; idy++)
      for (size_t idz = 0; idz < bufa.shape[2]; idz++)
        pa(idx,idy,idz) = val;
}

py::array_t<double> reconstruction( py::array_t<double> A,
                                    py::array_t<double> B,
                                    py::array_t<double> nnf,
                                    int patch_size )
{
    // allocate buffer for reconstruction
    auto bufA = A.request(), bufB = B.request(), bufnnf = nnf.request();
    auto reconsturcted = py::array_t<double>(bufA.shape);
    auto weights = py::array_t<double>(bufA.shape);

    // pointers
    auto prec = reconsturcted.mutable_unchecked<3>();
    auto pnnf = nnf.mutable_unchecked<3>();
    auto pweights = weights.mutable_unchecked<3>();
    auto pB = B.mutable_unchecked<3>();

    // dimensions
    int nnf_h = bufnnf.shape[0], nnf_w = bufnnf.shape[1];
    int A_h = bufA.shape[0], A_w = bufA.shape[1];
    int B_h = bufB.shape[0], B_w = bufB.shape[1];
    int channels = bufB.shape[2];

    int i1,j1,nnf_i1, nnf_j1;
    float nnf_i, nnf_j;

    put_val(reconsturcted,0.0);
    put_val(weights,1.0);

    int half_patch_size = static_cast <int> (floor(patch_size/2.0));
    for (int i = half_patch_size + 1; i < nnf_h; i++)
      for (int j =half_patch_size + 1; j < nnf_w; j++) {
        nnf_i = pnnf(i,j,0);
        nnf_j = pnnf(i,j,1);
        for (int k = -half_patch_size; k<1; k++)
          for (int l = -half_patch_size; l<1; l++) {
            i1 = i+k, j1 = j+l;
            nnf_i1 = static_cast <int> (nnf_i) + l;
            nnf_j1 = static_cast <int> (nnf_j) + k;
            if (is_clamped( B_h,B_w,
                            nnf_i1, nnf_i1+patch_size,
                            nnf_j1, nnf_j1+patch_size )) {
              // copy patch and increment weights
              for (size_t idx = 0; idx < patch_size; idx++)
                for (size_t idy = 0; idy < patch_size; idy++)
                  for (size_t idz = 0; idz < channels; idz++) {
                    prec(i1+idx,j1+idy,idz) += pB(nnf_i1+idx,nnf_j1+idy,idz);
                    pweights(i1+idx,j1+idy,idz) += 1.0;
                  }
            }
          }
      }
      for (size_t idx = 0; idx < A_h; idx++)
        for (size_t idy = 0; idy < A_w; idy++)
          for (size_t idz = 0; idz < channels; idz++)
            prec(idx,idy,idz) /= pweights(idx,idy,idz);
      return reconsturcted;
}


py::array_t<double> nnf_approx( py::array_t<double> A,
                                py::array_t<double> B,
                                py::array_t<double> nnf,
                                int patch_size, int iterations) {
  std::srand(std::time(nullptr));
  initialization(A,B,nnf,patch_size);
  for (int i=0; i<iterations; i++) {
    std::cout << "iteration: " << i+1 <<'\n';
    iteration(A,B,nnf,patch_size,(i+1)%2==0,1);
  }
  return reconstruction(A,B,nnf,patch_size);
}

PYBIND11_MODULE(patchmatch, m) {
    m.def("nnf_approx", &nnf_approx, "nnf_approx");
}
/*
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` patchmatch.cpp -o patchmatch`python3-config --extension-suffix`
*/
