//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch;

static bool haveSameSizeAndElementType(TensorType lhs, TensorType rhs) {
  if (lhs.hasRank() != rhs.hasRank())
    return false;
  bool sameSize = lhs.hasRank() ? lhs.getShape().equals(rhs.getShape()) : true;
  bool sameElementType = lhs.getElementType() == rhs.getElementType();
  return sameElementType && sameSize;
}

//===----------------------------------------------------------------------===//
// ToBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ToBuiltinTensorOp::verify() {
  auto resultType = getResult().getType().cast<TensorType>();
  auto operandType =
      getOperand().getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  if (!haveSameSizeAndElementType(resultType, operandType)) {
    return emitError()
           << "operand and result must have the same size and dtype";
  }
  return success();
}

LogicalResult ToBuiltinTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType =
      operands[0].getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  if (!resultType)
    return failure();
  inferredReturnTypes.push_back(resultType);
  return success();
}

void ToBuiltinTensorOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add(+[](ToBuiltinTensorOp op, PatternRewriter &rewriter) {
    auto fromBuiltinTensorOp =
        op.getOperand().getDefiningOp<FromBuiltinTensorOp>();
    if (!fromBuiltinTensorOp)
      return rewriter.notifyMatchFailure(op, "operand not FromBuiltinTensorOp");
    rewriter.replaceOp(op, fromBuiltinTensorOp.getOperand());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// FromBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult FromBuiltinTensorOp::verify() {
  auto resultType =
      getResult().getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  auto operandType = getOperand().getType().cast<TensorType>();
  if (!haveSameSizeAndElementType(resultType, operandType)) {
    return emitError()
           << "operand and result must have the same size and dtype";
  }
  return success();
}

void FromBuiltinTensorOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(+[](FromBuiltinTensorOp op, PatternRewriter &rewriter) {
    auto toBuiltinTensorOp = op.getOperand().getDefiningOp<ToBuiltinTensorOp>();
    if (!toBuiltinTensorOp)
      return rewriter.notifyMatchFailure(op, "operand not ToBuiltinTensorOp");
    rewriter.replaceOp(op, toBuiltinTensorOp.getOperand());
    return success();
  });
}

//===----------------------------------------------------------------------===//
// FromI64Op
//===----------------------------------------------------------------------===//

OpFoldResult FromI64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// ToI64Op
//===----------------------------------------------------------------------===//

OpFoldResult ToI64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// FromI1Op
//===----------------------------------------------------------------------===//

OpFoldResult FromI1Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// ToI1Op
//===----------------------------------------------------------------------===//

OpFoldResult ToI1Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// ToF64Op
//===----------------------------------------------------------------------===//

OpFoldResult ToF64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::FloatAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// FromF64Op
//===----------------------------------------------------------------------===//

OpFoldResult FromF64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::FloatAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.cpp.inc"
