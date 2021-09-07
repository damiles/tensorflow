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

#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Support/LogicalResult.h"                  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/constant_utils.h"

namespace mlir {
namespace TFL {

// TODO Make sure it covers all cases.
bool isFloatReciprocal(DivOp op) {
  auto numerator_qconst_op = op.lhs().getDefiningOp<ConstantOp>();
  if (!numerator_qconst_op) {
    return false;
  }
  auto numerators =
      numerator_qconst_op.value().dyn_cast_or_null<DenseFPElementsAttr>();
  if (!numerators || numerators.getNumElements() == 0 ||
      !numerators.isSplat()) {
    return false;
  }

  // We know that numerators is a non-empty splat, just take the first value
  const APFloat numerator = *numerators.begin();
  if (numerators.getType().getElementType().isF32() &&
      numerator == APFloat(1.0f)) {
    return true;
  }

  if (numerators.getType().getElementType().isF64() &&
      numerator == APFloat(1.0)) {
    return true;
  }

  return false;
}

// Lower TFL::DivOp x/y to x*1/y if inference_type is QINT8 or QINT16.
struct DivToMulReciprocal : public OpRewritePattern<DivOp> {
  explicit DivToMulReciprocal(MLIRContext* context,
                              tensorflow::DataType inference_type)
      : OpRewritePattern<DivOp>(context, 1), inference_type_(inference_type) {}
  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter& rewriter) const override {
    if (inference_type_ != tensorflow::DataType::DT_QINT8 &&
        inference_type_ != tensorflow::DataType::DT_QINT16) {
      return failure();
    }

    if (isFloatReciprocal(op)) {
      return failure();
    }

    auto rhs_tensor = op.rhs().getType().dyn_cast<RankedTensorType>();
    if (!rhs_tensor || !rhs_tensor.getElementType().isa<FloatType>()) {
      return failure();
    }

    auto lhs_tensor = op.lhs().getType().dyn_cast<RankedTensorType>();
    if (!lhs_tensor ||
        lhs_tensor.getElementType() != rhs_tensor.getElementType()) {
      return failure();
    }

    auto loc = op.getLoc();
    auto one_const_op = CreateConstOpWithSingleValue(
        &rewriter, loc, RankedTensorType::get({1}, rhs_tensor.getElementType()),
        1);
    auto reciprocal_op = rewriter.create<DivOp>(
        loc, op.rhs().getType(), *one_const_op, op.rhs(), "NONE");
    assert(isFloatReciprocal(reciprocal_op));

    auto mul_op = rewriter.create<MulOp>(loc, op.getResult().getType(),
                                         op.lhs(), reciprocal_op.getResult(),
                                         op.fused_activation_function());
    rewriter.replaceOp(op, mul_op.getResult());

    return success();
  }

 private:
  tensorflow::DataType inference_type_;
};

struct DivToMulReciprocalPass
    : public PassWrapper<DivToMulReciprocalPass, FunctionPass> {
  explicit DivToMulReciprocalPass(const QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {}

  void runOnFunction() override {
    auto func = getFunction();
    auto* ctx = func.getContext();
    OwningRewritePatternList patterns(ctx);
    patterns.insert<DivToMulReciprocal>(ctx, quant_specs_.inference_type);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
  }

 private:
  QuantizationSpecs quant_specs_;
};

std::unique_ptr<OperationPass<FuncOp>> CreateDivToMulReciprocalPass(
    const QuantizationSpecs& quant_specs) {
  return std::make_unique<DivToMulReciprocalPass>(quant_specs);
}

}  // namespace TFL
}  // namespace mlir
