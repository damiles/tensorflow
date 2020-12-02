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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_TO_TABLE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_TO_TABLE_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"              // from @llvm-project
#include "mlir/IR/PatternMatch.h"             // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

void populateWithLowerToTable(MLIRContext* context,
                              OwningRewritePatternList& patterns);

struct LowerExpToTable : public OpRewritePattern<ExpOp> {
  explicit LowerExpToTable(MLIRContext* context)
      : OpRewritePattern<ExpOp>(context, 1) {}

  LogicalResult matchAndRewrite(ExpOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_TO_TABLE_H_
