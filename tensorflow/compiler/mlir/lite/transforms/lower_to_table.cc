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

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"             // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace mlir {
namespace TFL {

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

  const int lut_type_int_width =
      std::numeric_limits<typename std::make_unsigned<LutOutT>::type>::digits;
  const bool lut_type_is_signed = std::is_signed<LutOutT>::value;
  auto lut_type = RankedTensorType::get(
      {lut.size()},
      rewriter.getIntegerType(lut_type_int_width, lut_type_is_signed));
  auto attr = DenseElementsAttr::get<LutOutT>(lut_type, lut);
  auto const_op = rewriter.create<ConstantOp>(original_op_loc, lut_type, attr);

  return const_op;
}

ConstantOp getLUT(PatternRewriter& rewriter, Location original_op_loc,
                  const mlir::quant::UniformQuantizedType& in_quant,
                  const mlir::quant::UniformQuantizedType& out_quant,
                  const std::function<double(double)>& lut_gen_func) {
  if (!in_quant.isSigned() || !out_quant.isSigned()) {
    return ConstantOp();
  } else if (in_quant.getStorageTypeIntegralWidth() == 8 &&
             out_quant.getStorageTypeIntegralWidth() == 8) {
    return getLUT<int8_t, int8_t>(rewriter, original_op_loc, in_quant,
                                  out_quant, lut_gen_func);
  } else if (in_quant.getStorageTypeIntegralWidth() == 16 &&
             out_quant.getStorageTypeIntegralWidth() == 16) {
    return getLUT<int16_t, int16_t>(rewriter, original_op_loc, in_quant,
                                    out_quant, lut_gen_func);
  } else {
    return ConstantOp();
  }
}

// Lower int8 and int16 TFL::ExpOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerExpToTable : public OpRewritePattern<ExpOp> {
  explicit LowerExpToTable(MLIRContext* context)
      : OpRewritePattern<ExpOp>(context, 1) {}

  LogicalResult matchAndRewrite(ExpOp op,
                                PatternRewriter& rewriter) const override {
    auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double exp_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    auto exp_lut_gen_func = [&](double value) {
      return std::exp(value) + exp_lut_offset;
    };

    auto const_op =
        getLUT(rewriter, op.getLoc(), in_quant, out_quant, exp_lut_gen_func);
    if (!const_op) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TableOp>(op, op.getResult().getType(), op.x(),
                                         const_op);

    return success();
  }
};

// Lower int8 and int16 TFL::LogOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerLogToTable : public OpRewritePattern<LogOp> {
  explicit LowerLogToTable(MLIRContext* context)
      : OpRewritePattern<LogOp>(context, 1) {}

  LogicalResult matchAndRewrite(LogOp op,
                                PatternRewriter& rewriter) const override {
    auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double output_min =
        out_quant.getScale() *
        (out_quant.getStorageTypeMin() - out_quant.getZeroPoint());
    const double log_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    auto log_lut_gen_func = [&](double value) {
      const double log_val = (value <= 0.0) ? output_min : std::log(value);
      return log_val + log_lut_offset;
    };

    auto const_op =
        getLUT(rewriter, op.getLoc(), in_quant, out_quant, log_lut_gen_func);
    if (!const_op) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TableOp>(op, op.getResult().getType(), op.x(),
                                         const_op);

    return success();
  }
};

// Lower int8 and int16 TFL::SqrtOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerSqrtToTable : public OpRewritePattern<SqrtOp> {
  explicit LowerSqrtToTable(MLIRContext* context)
      : OpRewritePattern<SqrtOp>(context, 1) {}

  LogicalResult matchAndRewrite(SqrtOp op,
                                PatternRewriter& rewriter) const override {
    auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double sqrt_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    auto sqrt_lut_gen_func = [&](double value) {
      const double sqrt_val = (value <= 0.0) ? 0.0 : std::sqrt(value);
      return sqrt_val + sqrt_lut_offset;
    };

    auto const_op =
        getLUT(rewriter, op.getLoc(), in_quant, out_quant, sqrt_lut_gen_func);
    if (!const_op) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TableOp>(op, op.getResult().getType(), op.x(),
                                         const_op);

    return success();
  }
};

// Lower int8 and int16 TFL::RsqrtOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerRsqrtToTable : public OpRewritePattern<RsqrtOp> {
  explicit LowerRsqrtToTable(MLIRContext* context)
      : OpRewritePattern<RsqrtOp>(context, 1) {}

  LogicalResult matchAndRewrite(RsqrtOp op,
                                PatternRewriter& rewriter) const override {
    auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double output_max =
        out_quant.getScale() *
        (out_quant.getStorageTypeMax() - out_quant.getZeroPoint());
    const double rsqrt_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    auto rsqrt_lut_gen_func = [&](double value) {
      const double rsqrt_val =
          (value <= 0.0) ? output_max : 1.0 / std::sqrt(value);
      return rsqrt_val + rsqrt_lut_offset;
    };

    auto const_op =
        getLUT(rewriter, op.getLoc(), in_quant, out_quant, rsqrt_lut_gen_func);
    if (!const_op) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TableOp>(op, op.getResult().getType(), op.x(),
                                         const_op);

    return success();
  }
};

// Lower int8 and int16 TFL::ExpOp to a TFL::TableOp based on the input and
// output scale and zero-point if the exponent is a constant.
struct LowerPowWithConstExponentToTable : public OpRewritePattern<PowOp> {
  LowerPowWithConstExponentToTable(MLIRContext* context)
      : OpRewritePattern<PowOp>(context, 1) {}

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs_quant = getUniformQuantizeType(op.lhs().getType());
    auto rhs_quant = getUniformQuantizeType(op.rhs().getType());
    auto out_quant = getUniformQuantizeType(op.output().getType());
    if (!lhs_quant || !rhs_quant || !out_quant) {
      return failure();
    }

    auto qconst_op = op.rhs().getDefiningOp<QConstOp>();
    if (!qconst_op) {
      return failure();
    }

    auto values = qconst_op.value().dyn_cast_or_null<DenseIntElementsAttr>();
    if (!values || values.getNumElements() == 0 || !values.isSplat()) {
      return failure();
    }

    // We know that values is a non-empty splat, just take the first value
    const APInt int_exponent = *values.begin();
    const double exponent =
        rhs_quant.getScale() *
        (int_exponent.getSExtValue() - rhs_quant.getZeroPoint());

    const double pow_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    auto pow_lut_gen_func = [&](double value) {
      return std::pow(value, exponent) + pow_lut_offset;
    };

    auto const_op =
        getLUT(rewriter, op.getLoc(), lhs_quant, out_quant, pow_lut_gen_func);
    if (!const_op) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TableOp>(op, op.output().getType(), op.lhs(),
                                         const_op);

    return success();
  }
};

struct LowerToTable : public PassWrapper<LowerToTable, FunctionPass> {
  LowerToTable() = default;

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    auto* ctx = func.getContext();
    patterns.insert<LowerExpToTable>(ctx);
    patterns.insert<LowerLogToTable>(ctx);
    patterns.insert<LowerSqrtToTable>(ctx);
    patterns.insert<LowerRsqrtToTable>(ctx);
    patterns.insert<LowerPowWithConstExponentToTable>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<FuncOp>> CreateLowerToTablePass() {
  return std::make_unique<LowerToTable>();
}

}  // namespace TFL
}  // namespace mlir
