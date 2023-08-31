/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_UTILS_H_

#include <numeric>

namespace tflite {
namespace resize_utils {

enum class ResizeMode { BILINEAR, NEAREST };

// Align corners sets the scaling ratio to (OH - 1)/(IH - 1)
// rather than OH / IH. Similarly for width.
inline void PreprocessResizeParameters(int input_size, int output_size,
                                       ResizeMode resize_mode,
                                       bool align_corners,
                                       bool half_pixel_centers, int* scale_n,
                                       int* scale_d, int* offset) {
  // Dimension is length 1, we are just sampling from one value.
  if (input_size == 1) {
    *scale_n = output_size;
    *scale_d = 1;
    *offset = 0;
    return;
  }

  // Apply if aligned and capable to be aligned.
  const bool apply_aligned = align_corners && (output_size > 1);
  *scale_n = apply_aligned ? (output_size - 1) : output_size;
  *scale_d = apply_aligned ? (input_size - 1) : input_size;

  // Simplify the scalers, make sure they are even values.
  const int gcd = std::gcd(*scale_n, *scale_d);
  *scale_n = 2 * (*scale_n / gcd);
  *scale_d = 2 * (*scale_d / gcd);

  // If half pixel centers we need to sample half a pixel inward.
  *offset = half_pixel_centers ? *scale_d / 2 : 0;
  // If nearest neighbours we need to guarantee we round up.
  if (resize_mode == ResizeMode::NEAREST && align_corners) {
    *offset += *scale_n / 2;
  }
  if (resize_mode == ResizeMode::BILINEAR && half_pixel_centers) {
    *offset -= *scale_n / 2;
  }

  // Reduce the scaling ratio if possible, we know n and d are even
  if ((*offset & 1) == 0) {
    *scale_n /= 2;
    *scale_d /= 2;
    *offset /= 2;
  }
}
}  // namespace resize_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_UTILS_H_