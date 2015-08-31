#pragma once
#include <caffe/data_layers.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#ifdef WIN32
#define USE_PARALLEL
#ifdef USE_PARALLEL
#include <ppl.h>
using namespace Concurrency;
#endif
#endif
namespace caffe {
  template <typename Dtype>
  class ImageFeatureDataLayer : public BaseDataLayer<Dtype> {
  public:
    explicit ImageFeatureDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
    
    //virtual ~ImageFeatureDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void AddImageFeatureData(const std::list<cv::Mat>& src, const std::list<std::vector<float> >& features);
    virtual void AddImageFeatureData(const std::list<cv::Mat>& src);

    /*virtual inline LayerParameter_LayerType type() const {
      return LayerParameter_LayerType_IMAGE_FEATURE_DATA;
    }*/
    virtual inline const char* type() const { return "ImageFeatureData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
    {
      return Forward_cpu(bottom, top);
    }
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}


    //std::vector<std::string> hdf_filenames_;
    //unsigned int num_files_;
    //unsigned int current_file_;
    hsize_t current_row_;
    Blob<Dtype> data_blob_;
    Blob<Dtype> label_blob_;
    std::auto_ptr<Dtype> data_;
    //DISABLE_COPY_AND_ASSIGN(ImageFeatureDataLayer);
#ifdef USE_PARALLEL
    critical_section mutex;
#endif
  };
}