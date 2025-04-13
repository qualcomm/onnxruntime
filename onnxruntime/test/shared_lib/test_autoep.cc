// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <absl/base/config.h>
#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test_allocator.h"
#include "utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

template <typename ModelOutputT, typename ModelInputT = float, typename InputT = Input<float>>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::string& ep_to_select,
                          std::optional<std::filesystem::path> library_path,
                          const std::vector<InputT>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<ModelOutputT>& expected_values_y,
                          bool test_session_creation_only = false) {
  Ort::SessionOptions session_options;

  // manually specify EP to select for now
  Ort::GetApi().AddSessionConfigEntry(session_options, "test.ep_to_select", ep_to_select.c_str());
  // Ort::GetApi().SessionOptionsSetEpSelectionPolicy(session_options, policy, nullptr);

  if (library_path) {
    // use EP name as registration name for now. there's some hardcoded matching of names to special case
    // the provider bridge EPs short term.
    OrtEnv* c_api_env = env;
    Ort::GetApi().GetEpApi()->RegisterExecutionProviderLibrary(c_api_env, ep_to_select.c_str(),
                                                               library_path->c_str());
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  // caller wants to test running the model (not just loading the model)
  if (!test_session_creation_only) {
    auto default_allocator = std::make_unique<MockedOrtAllocator>();
    RunSession<ModelOutputT, ModelInputT, InputT>(default_allocator.get(),
                                                  session,
                                                  inputs,
                                                  output_name,
                                                  expected_dims_y,
                                                  expected_values_y,
                                                  nullptr);
  }
}

TEST(AutoEpSelection, CpuEP) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       std::string(kCpuExecutionProvider), std::nullopt,
                       inputs, "Y", expected_dims_y, expected_values_y);
}

#if defined(USE_CUDA)
TEST(AutoEpSelection, CudaEP) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       std::string(kCudaExecutionProvider), "onnxruntime_providers_cuda",
                       inputs, "Y", expected_dims_y, expected_values_y);
}
#endif
#if defined(USE_DML)
TEST(AutoEpSelection, DmlEP) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       std::string(kDmlExecutionProvider), std::nullopt,
                       inputs, "Y", expected_dims_y, expected_values_y);
}
#endif

#if defined(USE_WEBGPU)
TEST(AutoEpSelection, WebGpuEP) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       std::string(kWebGpuExecutionProvider), std::nullopt,
                       inputs, "Y", expected_dims_y, expected_values_y);
}
#endif

}  // namespace test
}  // namespace onnxruntime
