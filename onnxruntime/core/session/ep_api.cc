// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include "core/framework/error_code_helper.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;
namespace OrtExecutionProviderApi {
ORT_API_STATUS_IMPL(SessionOptions_GetConfigOptions, _In_ const OrtSessionOptions* session_options,
                    _Out_ OrtKeyValuePairs** options) {
  API_IMPL_BEGIN
  // prefer existing_value if set (which should only be the case inside CreateEp)
  const auto* value = session_options->existing_value.value_or(&session_options->value);

  OrtKeyValuePairs* kvps = nullptr;
  OrtApis::CreateKeyValuePairs(&kvps);
  kvps->Copy(value->config_options.configurations);

  *options = kvps;

  return nullptr;
  API_IMPL_END
}

// Get ConfigOptions by key. Returns null in value if key not found (vs pointer to empty string if found).
ORT_API(const char*, SessionOptions_GetConfigOption, _In_ const OrtSessionOptions* session_options,
        _In_ const char* key) {
  // prefer existing_value if set (which should only be the case inside CreateEp)
  const auto* value = session_options->existing_value.value_or(&session_options->value);
  const auto& entries = value->config_options.configurations;
  if (auto it = entries.find(key), end = entries.end(); it != end) {
    return it->second.c_str();
  }

  return nullptr;
}

ORT_API(GraphOptimizationLevel, SessionOptions_GetOptimizationLevel, _In_ const OrtSessionOptions* session_options) {
  // prefer existing_value if set (which should only be the case inside CreateEp)
  const auto* value = session_options->existing_value.value_or(&session_options->value);

  switch (value->graph_optimization_level) {
    case onnxruntime::TransformerLevel::Default:
      return ORT_DISABLE_ALL;
    case onnxruntime::TransformerLevel::Level1:
      return ORT_ENABLE_BASIC;
    case onnxruntime::TransformerLevel::Level2:
      return ORT_ENABLE_EXTENDED;
    case onnxruntime::TransformerLevel::MaxLevel:
      return ORT_ENABLE_ALL;
    default:
      // This should never happen, but if it does, we return ORT_DISABLE_ALL to be safe.
      return ORT_DISABLE_ALL;
  }
}

static constexpr OrtEpApi ort_ep_api = {
    // NOTE: The C# bindings depend on the order within this struct, so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtExecutionProviderApi::SessionOptions_GetConfigOptions,
    &OrtExecutionProviderApi::SessionOptions_GetConfigOption,
    &OrtExecutionProviderApi::SessionOptions_GetOptimizationLevel,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, SessionOptions_GetOptimizationLevel) / sizeof(void*) == 2,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}

}  // namespace OrtExecutionProviderApi
