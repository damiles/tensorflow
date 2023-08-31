/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void ComputeInterpolationValues(const float value, const float scale,
                                       const bool half_pixel_centers,
                                       int32_t input_size, float* scaled_value,
                                       int32_t* lower_bound,
                                       int32_t* upper_bound) {
  if (half_pixel_centers) {
    *scaled_value = (value + 0.5f) * scale - 0.5f;
  } else {
    *scaled_value = value * scale;
  }
  float scaled_value_floor = std::floor(*scaled_value);
  *lower_bound = std::max(static_cast<int32_t>(scaled_value_floor),
                          static_cast<int32_t>(0));
  *upper_bound =
      std::min(static_cast<int32_t>(std::ceil(*scaled_value)), input_size - 1);
}

template <typename T>
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const T* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32_t* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  int32_t output_height =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  int32_t output_width =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }
  const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32_t y0, y1;
      ComputeInterpolationValues(y, height_scale, op_params.half_pixel_centers,
                                 input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32_t x0, x1;
        ComputeInterpolationValues(x, width_scale, op_params.half_pixel_centers,
                                   input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_shape, b, y0, x0, c)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y1, x0, c)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y0, x1, c)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_shape, b, y1, x1, c)] *
                                 (input_y - y0) * (input_x - x0) +
                             rounding_offset);
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

template <typename T>
inline void ResizeBilinearInteger(
    const tflite::ResizeBilinearParams& op_params,
    const RuntimeShape& unextended_input_shape, const T* input_data,
    const RuntimeShape& unextended_output_size_shape,
    const int32_t* output_size_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  static constexpr int32_t kMinOutput = std::numeric_limits<T>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<T>::max();

  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t input_height = input_shape.Dims(1);
  const int32_t input_width = input_shape.Dims(2);
  const int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  const int32_t output_height =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  const int32_t output_width =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      const int64_t sy = y * op_params.scale_y_d + op_params.offset_y;
      const int64_t iy = IntFloorDiv(sy, op_params.scale_y_n);
      const int64_t ry = sy - iy * op_params.scale_y_n;
      const int32_t iy0 = std::max<int32_t>(iy, 0);
      const int32_t iy1 =
          static_cast<int32_t>(std::min<int64_t>(iy + 1, input_height - 1));

      for (int x = 0; x < output_width; ++x) {
        const int64_t sx = x * op_params.scale_x_d + op_params.offset_x;
        const int64_t ix = IntFloorDiv(sx, op_params.scale_x_n);
        const int64_t rx = sx - ix * op_params.scale_x_n;
        const int32_t ix0 = std::max<int32_t>(ix, 0);
        const int32_t ix1 =
            static_cast<int32_t>(std::min<int64_t>(ix + 1, input_width - 1));

        for (int c = 0; c < depth; ++c) {
          const T v00 = input_data[Offset(input_shape, b, iy0, ix0, c)];
          const T v01 = input_data[Offset(input_shape, b, iy0, ix1, c)];
          const T v10 = input_data[Offset(input_shape, b, iy1, ix0, c)];
          const T v11 = input_data[Offset(input_shape, b, iy1, ix1, c)];

          int64_t acc = 0;
          acc += v00 * (op_params.scale_y_n - ry) * (op_params.scale_x_n - rx);
          acc += v01 * (op_params.scale_y_n - ry) * rx;
          acc += v10 * ry * (op_params.scale_x_n - rx);
          acc += v11 * ry * rx;

          const int32_t interpolated = MultiplyByQuantizedMultiplier(
              acc, op_params.output_multiplier, op_params.output_shift);
          const int32_t clamped_interpolated =
              std::max(std::min(interpolated, kMaxOutput), kMinOutput);
          output_data[Offset(output_shape, b, y, x, c)] = clamped_interpolated;
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_
