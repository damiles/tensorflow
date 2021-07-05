/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/lower_to_table.h"

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"             // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace mlir {
namespace TFL {

void populateWithLowerToTable(MLIRContext* context,
                              OwningRewritePatternList& patterns) {
  patterns.insert<LowerExpToTable>(context);
}

mlir::quant::UniformQuantizedType getUniformQuantizeType(mlir::Type type) {
  auto quant_type = quant::UniformQuantizedType::getQuantizedElementType(type);
  if (!quant_type) {
    return mlir::quant::UniformQuantizedType();
  }

  auto uquant_type = quant_type.dyn_cast<mlir::quant::UniformQuantizedType>();
  if (!uquant_type) {
    return mlir::quant::UniformQuantizedType();
  }

  return uquant_type;
}

template <typename LutInT, typename LutOutT>
ConstantOp getLUT(PatternRewriter& rewriter, Location original_op_loc,
                  const mlir::quant::UniformQuantizedType& in_quant,
                  const mlir::quant::UniformQuantizedType& out_quant,
                  const std::function<double(double)>& lut_gen_func) {
  const double input_min = in_quant.getScale() * (in_quant.getStorageTypeMin() -
                                                  in_quant.getZeroPoint());
  const double input_max = in_quant.getScale() * (in_quant.getStorageTypeMax() -
                                                  in_quant.getZeroPoint());

  const double output_min =
      out_quant.getScale() *
      (out_quant.getStorageTypeMin() - out_quant.getZeroPoint());
  const double output_max =
      out_quant.getScale() *
      (out_quant.getStorageTypeMax() - out_quant.getZeroPoint());

  std::vector<LutOutT> lut(tflite::lut_size<LutInT>());
  tflite::gen_lut<double, LutInT, LutOutT>(lut_gen_func, input_min, input_max,
                                           output_min, output_max, lut.data());

  auto type = RankedTensorType::get(
      {lut.size()},
      rewriter.getIntegerType(
          std::numeric_limits<
              typename std::make_unsigned<LutOutT>::type>::digits));
  auto attr = DenseElementsAttr::get<LutOutT>(type, lut);
  auto const_op = rewriter.create<ConstantOp>(original_op_loc, type, attr);

  return const_op;
}

LogicalResult LowerExpToTable::matchAndRewrite(
    ExpOp op, PatternRewriter& rewriter) const {
  auto in_quant = getUniformQuantizeType(op.getOperand().getType());
  auto out_quant = getUniformQuantizeType(op.getResult().getType());
  if (!in_quant || !out_quant) {
    return failure();
  }

  const double exp_lut_offset = out_quant.getZeroPoint() * out_quant.getScale();
  auto exp_lut_gen_func = [&](double value) {
    return std::exp(value) + exp_lut_offset;
  };

  ConstantOp table;
  if (in_quant.isSigned() && out_quant.isSigned() &&
      in_quant.getStorageTypeIntegralWidth() == 8 &&
      out_quant.getStorageTypeIntegralWidth() == 8) {
    table = getLUT<int8_t, int8_t>(rewriter, op.getLoc(), in_quant, out_quant,
                                   exp_lut_gen_func);
  } else {
    return failure();
  }

  ShapedType type = RankedTensorType::get({0}, rewriter.getIntegerType(8, false));
  OpaqueElementsAttr custom_option = OpaqueElementsAttr::get(
      rewriter.getContext()->getLoadedDialect("tfl"), type, StringRef());

  rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
      op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
      "Table", custom_option);
  return success();
}

}  // namespace TFL
}  // namespace mlir
