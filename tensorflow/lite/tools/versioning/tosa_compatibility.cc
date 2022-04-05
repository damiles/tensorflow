/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/tosa_compatibility.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {

absl::Status CheckTOSACompatibility(const OpSignature& op_sig) {
  TfLiteBuiltinOperator opcode = static_cast<TfLiteBuiltinOperator>(op_sig.op);
  switch (opcode) {
    case kTfLiteBuiltinConv2d: {
      return absl::OkStatus();
    }
    case kTfLiteBuiltinFullyConnected: {
      return absl::OkStatus();
    }
    case kTfLiteBuiltinReshape: {
      return absl::OkStatus();
    }

    default:
      break;
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Not supported op ", tflite::EnumNamesBuiltinOperator()[op_sig.op]));
}

absl::Status CheckTOSACompatibility(const OperatorCode* op_code,
                                    const Operator* op,
                                    const SubGraph* subgraph,
                                    const Model* model) {
  OpSignature op_sig = GetOpSignature(op_code, op, subgraph, model);
  auto status = CheckTOSACompatibility(op_sig);
  if (op_sig.builtin_data) {
    free(op_sig.builtin_data);
  }
  return status;
}

}  // namespace tflite
