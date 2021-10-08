//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "Pass/Passes.h"

#include "Dialect/PIM/IR/PIMOps.hpp"
#include "Dialect/PNM/IR/PNMOps.hpp"

#include "Conversion/PIMToPNM/PIMToPNM.h"

#include <iostream>
namespace cl = llvm::cl;
using namespace mlir;

static llvm::cl::opt<std::string> input_filename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> output_filename("o",
    llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));
    
void initPasses() {
  mlir::registerPass("convert-pim",
      "pim",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertPIMToPNMPass();
      });
}
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::PIMOpsDialect>();
  registry.insert<mlir::PNMOpsDialect>();
  initPasses();
  
//  mlir::registerAllPasses();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR\n");

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  assert(file);

  auto output = mlir::openOutputFile(output_filename, &error_message);
  assert(output);
  
//  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "test optimizer driver\n", registry));
  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
      registry, 0, 0, 1, 0, true));
}
