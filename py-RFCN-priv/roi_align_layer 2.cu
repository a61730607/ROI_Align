#include <cfloat>
#include <algorithm>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"
using std::max;
using std::min;

namespace caffe {

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    const Dtype* bottom_rois, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;

    Dtype roi_width = max(roi_end_w - roi_start_w, (T)1.);
    Dtype roi_height = max(roi_end_h - roi_start_h, (T)1.);

    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    const Dtype* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    
    const Dtype count = roi_bin_grid_h * roi_bin_grid_w;

    Dtype output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const Dtype y = roi_start_h + ph * bin_size_h +
          static_cast<Dtype>(iy + .5f) * bin_size_h /
              static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const Dtype x = roi_start_w + pw * bin_size_w +
            static_cast<Dtype>(ix + .5f) * bin_size_w /
                static_cast<Dtype>(roi_bin_grid_w);

        Dtype val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
    
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  if (bottom.size() > 2) {
      const Dtype* scale_pred = bottom[2]->gpu_data();
      caffe_gpu_asum<Dtype>(1, scale_pred, &spatial_scale_);
  }
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, sampling_ratio_, bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}
template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename Dtype>
__global__ void ROIAlignBackward(
    const int nthreads, const Dtype* top_diff, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, 
    const int sampling_ratio,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)1.);
    Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1.);
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    Dtype* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* offset_top_diff = top_diff + top_offset;
    const Dtype top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const Dtype y = roi_start_h + ph * bin_size_h +
          static_cast<Dtype>(iy + .5f) * bin_size_h /
              static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const Dtype x = roi_start_w + pw * bin_size_w +
            static_cast<Dtype>(ix + .5f) * bin_size_w /
                static_cast<Dtype>(roi_bin_grid_w);

        Dtype w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          gpu_atomic_add(
              static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          gpu_atomic_add(
              static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          gpu_atomic_add(
              static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          gpu_atomic_add(
              static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        } // if
      } // ix
    } // iy
  } // CUDA_KERNEL_LOOP
} // RoIAlignBackward


template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  //auto& X = Input(0);
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  //auto& R = Input(1);
  const Dtype* top_diff = top[0]->gpu_diff();
  //auto& dY = Input(2);
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //auto* dX = Output(0);
  
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const int count = top[0]->count();

  if (bottom.size() > 2) {
      const Dtype* scale_pred = bottom[2]->gpu_data();
      caffe_gpu_asum<Dtype>(1, scale_pred, &spatial_scale_);
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff,  bottom[1]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, sampling_ratio_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}
