// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;
namespace OrtExecutionProviderApi {
ORT_API_STATUS_IMPL(RegisterExecutionProviderLibrary, _In_ OrtEnv* env, const char* registration_name,
                    const ORTCHAR_T* path) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().RegisterExecutionProviderLibrary(registration_name, path));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(UnregisterExecutionProviderLibrary, _In_ OrtEnv* env, const char* registration_name) {
  API_IMPL_BEGIN
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().UnregisterExecutionProviderLibrary(registration_name));
  return nullptr;
  API_IMPL_END
}

// OrtExecutionDevice accessors
ORT_API(OrtHardwareDeviceType, HardwareDevice_Type, _In_ const OrtHardwareDevice* device) {
  return OrtHardwareDeviceType(device->type);
}

ORT_API(int32_t, HardwareDevice_VendorId, _In_ const OrtHardwareDevice* device) {
  return device->vendor_id;
}

ORT_API(const char*, HardwareDevice_Vendor, _In_ const OrtHardwareDevice* device) {
  return device->vendor.c_str();
}

ORT_API(int32_t, HardwareDevice_BusId, _In_ const OrtHardwareDevice* device) {
  return device->bus_id;
}

ORT_API(const OrtKeyValuePairs*, HardwareDevice_Properties, _In_ const OrtHardwareDevice* device) {
  return &device->properties;
}

// ORT_API(const char*, ExecutionDevice_EpName, _In_ const OrtExecutionDevice* exec_device) {
//   return exec_device->ep_name.c_str();
// }
//
// ORT_API(const char*, ExecutionDevice_EpVendor, _In_ const OrtExecutionDevice* exec_device) {
//   return exec_device->ep_vendor.c_str();
// }
//
// ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpMetadata, _In_ const OrtExecutionDevice* exec_device) {
//   return &exec_device->ep_metadata;
// }
//
// ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpOptions, _In_ const OrtExecutionDevice* exec_device) {
//   return &exec_device->ep_options;
// }
//
// ORT_API(const OrtHardwareDevice*, ExecutionDevice_Device, _In_ const OrtExecutionDevice* exec_device) {
//   return exec_device->device;
// }

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

    &OrtExecutionProviderApi::RegisterExecutionProviderLibrary,
    &OrtExecutionProviderApi::UnregisterExecutionProviderLibrary,

    &OrtExecutionProviderApi::HardwareDevice_Type,
    &OrtExecutionProviderApi::HardwareDevice_VendorId,
    &OrtExecutionProviderApi::HardwareDevice_Vendor,
    &OrtExecutionProviderApi::HardwareDevice_BusId,
    &OrtExecutionProviderApi::HardwareDevice_Properties,

    //&OrtExecutionProviderApi::ExecutionDevice_EpName,
    //&OrtExecutionProviderApi::ExecutionDevice_EpVendor,
    //&OrtExecutionProviderApi::ExecutionDevice_EpMetadata,
    //&OrtExecutionProviderApi::ExecutionDevice_EpOptions,
    //&OrtExecutionProviderApi::ExecutionDevice_Device,

    &OrtExecutionProviderApi::SessionOptionsConfigOptions,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, SessionOptionsConfigOption) / sizeof(void*) == 8,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}

}  // namespace OrtExecutionProviderApi
