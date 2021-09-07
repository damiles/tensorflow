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

quant::UniformQuantizedType getUniformQuantizeType(Type type) {
  auto quant_type = quant::UniformQuantizedType::getQuantizedElementType(type);
  if (!quant_type) {
    return quant::UniformQuantizedType();
  }

  auto uquant_type = quant_type.dyn_cast<quant::UniformQuantizedType>();
  if (!uquant_type) {
    return quant::UniformQuantizedType();
  }

  return uquant_type;
}

template <typename LutInT, typename LutOutT>
ConstantOp getLUT(const Location& original_op_loc,
                  const quant::UniformQuantizedType& in_quant,
                  const quant::UniformQuantizedType& out_quant,
                  const std::function<double(double)>& lut_gen_func,
                  PatternRewriter& rewriter) {
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

ConstantOp getLUT(const Location& original_op_loc,
                  const quant::UniformQuantizedType& in_quant,
                  const quant::UniformQuantizedType& out_quant,
                  const std::function<double(double)>& lut_gen_func,
                  PatternRewriter& rewriter) {
  if (!in_quant.isSigned() || !out_quant.isSigned()) {
    return ConstantOp();
  } else if (in_quant.getStorageTypeIntegralWidth() == 8 &&
             out_quant.getStorageTypeIntegralWidth() == 8) {
    return getLUT<int8_t, int8_t>(original_op_loc, in_quant, out_quant,
                                  lut_gen_func, rewriter);
  } else if (in_quant.getStorageTypeIntegralWidth() == 16 &&
             out_quant.getStorageTypeIntegralWidth() == 16) {
    return getLUT<int16_t, int16_t>(original_op_loc, in_quant, out_quant,
                                    lut_gen_func, rewriter);
  } else {
    return ConstantOp();
  }
}

std::string getTableCustomCode() { return "Table"; }

OpaqueElementsAttr getTableCustomOption(PatternRewriter& rewriter) {
  return OpaqueElementsAttr::get(
      rewriter.getContext()->getLoadedDialect("tfl"),
      RankedTensorType::get({0}, rewriter.getIntegerType(8, false)),
      StringRef());
}

// Lower int8 and int16 TFL::ExpOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerExpToTable : public OpRewritePattern<ExpOp> {
  explicit LowerExpToTable(MLIRContext* context)
      : OpRewritePattern<ExpOp>(context, 1) {}

  LogicalResult matchAndRewrite(ExpOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double exp_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto exp_lut_gen_func = [&](double x) {
      return std::exp(x) + exp_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, exp_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
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
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double output_min =
        out_quant.getScale() *
        (out_quant.getStorageTypeMin() - out_quant.getZeroPoint());
    const double log_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    auto log_lut_gen_func = [&](double x) {
      const double log_x = (x <= 0.0) ? output_min : std::log(x);
      return log_x + log_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, log_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
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
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double sqrt_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto sqrt_lut_gen_func = [&](double x) {
      const double sqrt_x = (x <= 0.0) ? 0.0 : std::sqrt(x);
      return sqrt_x + sqrt_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, sqrt_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
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
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double output_max =
        out_quant.getScale() *
        (out_quant.getStorageTypeMax() - out_quant.getZeroPoint());
    const double rsqrt_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto rsqrt_lut_gen_func = [&](double x) {
      const double rsqrt_x = (x <= 0.0) ? output_max : 1.0 / std::sqrt(x);
      return rsqrt_x + rsqrt_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, rsqrt_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::EluOp to a TFL::TableOp based on the input
// and output scale and zero-point.
struct LowerEluToTable : public OpRewritePattern<EluOp> {
  explicit LowerEluToTable(MLIRContext* context)
      : OpRewritePattern<EluOp>(context, 1) {}

  LogicalResult matchAndRewrite(EluOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double elu_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto elu_lut_gen_func = [&](double x) {
      const double elu_x = x < 0.0 ? std::exp(x) - 1.0 : x;
      return elu_x + elu_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, elu_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::HardSwishOp to a TFL::TableOp based on the input
// and output scale and zero-point.
struct LowerHardSwishToTable : public OpRewritePattern<HardSwishOp> {
  explicit LowerHardSwishToTable(MLIRContext* context)
      : OpRewritePattern<HardSwishOp>(context, 1) {}

  LogicalResult matchAndRewrite(HardSwishOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double hard_swish_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto hard_swish_lut_gen_func = [&](double x) {
      // hard_swish(x) = x * (relu6(x + 3) / 6)
      double hard_swish_x;
      if (x <= -3.0) {
        hard_swish_x = 0.0;
      } else if (x >= 3.0) {
        hard_swish_x = x;
      } else {
        hard_swish_x = x * (x + 3.0) / 6.0;
      }
      return hard_swish_x + hard_swish_lut_offset;
    };

    ConstantOp table = getLUT(op.getLoc(), in_quant, out_quant,
                              hard_swish_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(),
        mlir::ArrayRef<Value>({op.input(), table}), getTableCustomCode(),
        getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::LogisticOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerLogisticToTable : public OpRewritePattern<LogisticOp> {
  explicit LowerLogisticToTable(MLIRContext* context)
      : OpRewritePattern<LogisticOp>(context, 1) {}

  LogicalResult matchAndRewrite(LogisticOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double logistic_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto logistic_lut_gen_func = [&](double x) {
      return 1.0 / (1.0 + std::exp(-x)) + logistic_lut_offset;
    };

    ConstantOp table = getLUT(op.getLoc(), in_quant, out_quant,
                              logistic_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::TanhOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerTanhToTable : public OpRewritePattern<TanhOp> {
  explicit LowerTanhToTable(MLIRContext* context)
      : OpRewritePattern<TanhOp>(context, 1) {}

  LogicalResult matchAndRewrite(TanhOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double tanh_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto tanh_lut_gen_func = [&](double x) {
      return std::tanh(x) + tanh_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, tanh_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(),
        mlir::ArrayRef<Value>({op.input(), table}), getTableCustomCode(),
        getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::CosOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerCosToTable : public OpRewritePattern<CosOp> {
  explicit LowerCosToTable(MLIRContext* context)
      : OpRewritePattern<CosOp>(context, 1) {}

  LogicalResult matchAndRewrite(CosOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double cos_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto cos_lut_gen_func = [&](double x) {
      return std::cos(x) + cos_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, cos_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::SinOp to a TFL::TableOp based on the input and
// output scale and zero-point.
struct LowerSinToTable : public OpRewritePattern<SinOp> {
  explicit LowerSinToTable(MLIRContext* context)
      : OpRewritePattern<SinOp>(context, 1) {}

  LogicalResult matchAndRewrite(SinOp op,
                                PatternRewriter& rewriter) const override {
    const auto in_quant = getUniformQuantizeType(op.getOperand().getType());
    const auto out_quant = getUniformQuantizeType(op.getResult().getType());
    if (!in_quant || !out_quant) {
      return failure();
    }

    const double sin_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto sin_lut_gen_func = [&](double x) {
      return std::sin(x) + sin_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), in_quant, out_quant, sin_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.getResult().getType(), mlir::ArrayRef<Value>({op.x(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::PowOp to a TFL::TableOp based on the input and
// output scale and zero-point if the exponent is a single-value constant.
struct LowerPowWithConstExponentToTable : public OpRewritePattern<PowOp> {
  LowerPowWithConstExponentToTable(MLIRContext* context)
      : OpRewritePattern<PowOp>(context, 1) {}
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto lhs_quant = getUniformQuantizeType(op.lhs().getType());
    const auto rhs_quant = getUniformQuantizeType(op.rhs().getType());
    const auto out_quant = getUniformQuantizeType(op.output().getType());
    if (!lhs_quant || !rhs_quant || !out_quant) {
      return failure();
    }

    auto exponent_qconst_op = op.rhs().getDefiningOp<QConstOp>();
    if (!exponent_qconst_op) {
      return failure();
    }

    const auto qint_exponents =
        exponent_qconst_op.value().dyn_cast_or_null<DenseIntElementsAttr>();
    const auto qfloat_exponents =
        exponent_qconst_op.value().dyn_cast_or_null<DenseFPElementsAttr>();

    double exponent;
    if (qint_exponents && qint_exponents.getNumElements() > 0 &&
        qint_exponents.isSplat()) {
      const APInt qexponent = *qint_exponents.begin();
      exponent = qexponent.getSExtValue();
    } else if (qfloat_exponents && qfloat_exponents.getNumElements() > 0 &&
               qfloat_exponents.isSplat()) {
      const APFloat qexponent = *qfloat_exponents.begin();
      exponent = qexponent.convertToDouble();
    } else {
      return failure();
    }

    const double scaled_exponent =
        rhs_quant.getScale() * (exponent - rhs_quant.getZeroPoint());
    const double pow_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto pow_lut_gen_func = [&](double value) {
      return std::pow(value, scaled_exponent) + pow_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), lhs_quant, out_quant, pow_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.output().getType(), mlir::ArrayRef<Value>({op.lhs(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

// Lower int8 and int16 TFL::DivOp with single const numerator to a TFL::TableOp
// based on the input and output scale and zero-point.
struct LowerDivWithConstNumeratorToTable : public OpRewritePattern<DivOp> {
  explicit LowerDivWithConstNumeratorToTable(MLIRContext* context)
      : OpRewritePattern<DivOp>(context, 1) {}
  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter& rewriter) const override {
    const auto lhs_quant = getUniformQuantizeType(op.lhs().getType());
    const auto rhs_quant = getUniformQuantizeType(op.rhs().getType());
    const auto out_quant = getUniformQuantizeType(op.output().getType());
    if (!lhs_quant || !rhs_quant || !out_quant) {
      return failure();
    }

    auto numerator_qconst_op = op.lhs().getDefiningOp<QConstOp>();
    if (!numerator_qconst_op) {
      return failure();
    }

    auto qnumerators =
        numerator_qconst_op.value().dyn_cast_or_null<DenseIntElementsAttr>();
    if (!qnumerators || qnumerators.getNumElements() == 0 ||
        !qnumerators.isSplat()) {
      return failure();
    }

    const APInt qnumerator = *qnumerators.begin();
    const double numerator = lhs_quant.getScale() * (qnumerator.getSExtValue() -
                                                     lhs_quant.getZeroPoint());
    const double output_max =
        out_quant.getScale() *
        (out_quant.getStorageTypeMax() - out_quant.getZeroPoint());
    const double div_lut_offset =
        out_quant.getZeroPoint() * out_quant.getScale();
    const auto div_lut_gen_func = [&](double value) {
      const double res = (value == 0.0) ? output_max : numerator / value;
      return res + div_lut_offset;
    };

    ConstantOp table =
        getLUT(op.getLoc(), rhs_quant, out_quant, div_lut_gen_func, rewriter);
    if (!table) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::TFL::CustomOp>(
        op, op.output().getType(), mlir::ArrayRef<Value>({op.rhs(), table}),
        getTableCustomCode(), getTableCustomOption(rewriter));
    return success();
  }
};

void populateWithLowerToTable(MLIRContext* context,
                              OwningRewritePatternList& patterns) {
  // TODO GELU with approximate=False to TABLE, need a TFL op for tf.math.erf
  patterns.insert<LowerExpToTable>(context);
  patterns.insert<LowerLogToTable>(context);
  patterns.insert<LowerSqrtToTable>(context);
  patterns.insert<LowerRsqrtToTable>(context);
  patterns.insert<LowerEluToTable>(context);
  patterns.insert<LowerHardSwishToTable>(context);
  patterns.insert<LowerLogisticToTable>(context);
  patterns.insert<LowerTanhToTable>(context);
  patterns.insert<LowerCosToTable>(context);
  patterns.insert<LowerSinToTable>(context);
  // TODO x**y could be decomposed in a general way into exp(ln(x)*y) for
  // positive x
  // https://en.wikipedia.org/wiki/Exponentiation#Powers_via_logarithms
  patterns.insert<LowerPowWithConstExponentToTable>(context);
  patterns.insert<LowerDivWithConstNumeratorToTable>(context);
}

}  // namespace TFL
}  // namespace mlir
