#pragma once
#include <string>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <vector>

class CAFFE_DLL_EXPORT GpuUtil
{
public:
  GpuUtil();
  ~GpuUtil();
  static std::vector<int> set_gpu(caffe::SolverParameter& solver_param, const std::string& gpu_param_str = "");
  static void RunSolver(caffe::Solver<float>* solver_ptr, std::vector<int>& gpus);
};

