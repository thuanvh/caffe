#include "GpuUtil.h"
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

// Parse GPU ids or use all available devices
void get_gpus(vector<int>* gpus, const std::string& gpu_param) {
  if (gpu_param == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  }
  else if (gpu_param.size()) {
    vector<string> strings;
    boost::split(strings, gpu_param, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  }
  else {
    CHECK_EQ(gpus->size(), 0);
  }
}

vector<int> GpuUtil::set_gpu(caffe::SolverParameter& solver_param, const std::string& gpu_param_str)
{
  string gpu_param = gpu_param_str;
  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (gpu_param.size() == 0
    && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    if (solver_param.has_device_id()) {
      gpu_param = "" +
        boost::lexical_cast<string>(solver_param.device_id());
    }
    else {  // Set default GPU if unspecified
      gpu_param = "" + boost::lexical_cast<string>(0);
    }
  }

  vector<int> gpus;
  get_gpus(&gpus, gpu_param);
  if (gpus.size() == 0) {
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();

    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }
  return gpus;
}

void GpuUtil::RunSolver(caffe::Solver<float>* solver_ptr, vector<int>& gpus)
{
  shared_ptr<caffe::Solver<float> >
    solver(solver_ptr);
  caffe::P2PSync<float> sync(solver, NULL, solver->param());
  sync.Run(gpus);
}

GpuUtil::GpuUtil()
{
}


GpuUtil::~GpuUtil()
{
}
