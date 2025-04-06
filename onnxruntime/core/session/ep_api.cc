// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include "core/framework/error_code_helper.h"
// #include "core/framework/ort_value.h"
// #include "core/framework/onnxruntime_typeinfo.h"
// #include "core/framework/tensor_type_and_shape.h"
// #include "core/graph/constants.h"
// #include "core/graph/model.h"
// #include "core/graph/model_editor_api_types.h"
// #include "core/graph/onnx_protobuf.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/abi_session_options_impl.h"
// #include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
// #include "core/session/utils.h"
#include "core/session/environment.h"

using namespace onnxruntime;
namespace OrtExecutionProviderApi {
ORT_API_STATUS_IMPL(RegisterEpFactory, OrtEnv* env, OrtEpApi::OrtEpFactory* factory) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().RegisterPluginEpFactory(*factory));
  return nullptr;
  API_IMPL_END
}
ORT_API_STATUS_IMPL(RegisterLegacyEpFactory, OrtEnv* env, const ORTCHAR_T* library_path) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().RegisterLegacyEpFactory(library_path));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateExecutionDevice,
                    _In_ /*const*/ OrtEpApi::OrtEp* ep,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_reads_(num_ep_device_properties) const char** ep_device_properties_keys,
                    _In_reads_(num_ep_device_properties) const char** ep_device_properties_values,
                    _In_ size_t num_ep_device_properties,
                    _Out_ OrtExecutionDevice** ort_execution_device) {
  API_IMPL_BEGIN
  auto ed = std::make_unique<OrtExecutionDevice>();
  ed->name = ep->GetName(ep);
  ed->execution_vendor = ep->GetVendor(ep);
  ed->device = hardware_device;
  // this makes the ownership semantics a little strange. who releases this memory?
  // would an opaque type work better?
  OrtApis::CreateKeyValuePairs(&ed->properties);
  for (size_t i = 0; i < num_ep_device_properties; ++i) {
    OrtApis::AddKeyValuePair(ed->properties, ep_device_properties_keys[i], ep_device_properties_values[i]);
  }

  *ort_execution_device = ed.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SessionOptionsConfigOptions, _In_ const OrtSessionOptions* session_options,
                    _Out_ OrtKeyValuePairs** options) {
  API_IMPL_BEGIN
  OrtKeyValuePairs* kvps = nullptr;
  OrtApis::CreateKeyValuePairs(&kvps);
  kvps->Copy(session_options->value.config_options.configurations);

  *options = kvps;

  return nullptr;
  API_IMPL_END
}

// Get ConfigOptions by key. Returns null in value if key not found (vs pointer to empty string if found).
ORT_API(const char*, SessionOptionsConfigOption, _In_ const OrtSessionOptions* session_options, _In_ const char* key) {
  const auto& entries = session_options->value.config_options.configurations;
  if (auto it = entries.find(key), end = entries.end(); it != end) {
    return it->second.c_str();
  }

  return nullptr;
}

static constexpr OrtEpApi ort_ep_api = {
    // NOTE: The C# bindings depend on the Api order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtExecutionProviderApi::RegisterEpFactory,
    &OrtExecutionProviderApi::RegisterLegacyEpFactory,
    &OrtExecutionProviderApi::CreateExecutionDevice,
    &OrtExecutionProviderApi::SessionOptionsConfigOptions,

    &OrtExecutionProviderApi::SessionOptionsConfigOption,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, SessionOptionsConfigOption) / sizeof(void*) == 4,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}

}  // namespace OrtExecutionProviderApi
