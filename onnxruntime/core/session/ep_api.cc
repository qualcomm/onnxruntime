// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include "core/common/basic_types.h"
#include "core/framework/error_code_helper.h"

// #include "core/framework/ort_value.h"
// #include "core/framework/onnxruntime_typeinfo.h"
// #include "core/framework/tensor_type_and_shape.h"
// #include "core/graph/constants.h"
// #include "core/graph/model.h"
// #include "core/graph/model_editor_api_types.h"
// #include "core/graph/onnx_protobuf.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/abi_session_options_impl.h"
// #include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
// #include "core/session/utils.h"
#include "core/session/environment.h"

using namespace onnxruntime;
namespace OrtExecutionProviderApi {
ORT_API_STATUS_IMPL(RegisterExecutionProviderLibrary, _In_ OrtEnv* env, const ORTCHAR_T* path, const char* ep_name) {
  API_IMPL_BEGIN
  // provider bridge EPs can provide path to library to load in session options.
  // might need logic to convert from session option ConfigParams to the EP specific settings
  //
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().RegisterExecutionProviderPluginLibrary(path, ep_name));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(UnregisterExecutionProviderLibrary, _In_ OrtEnv* env, const char* ep_name) {
  API_IMPL_BEGIN
  //
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().UnregisterExecutionProviderPluginLibrary(ep_name));
  return nullptr;
  API_IMPL_END
}

// OrtExecutionDevice accessors
ORT_API(OrtHardwareDeviceType, HardwareDevice_Type, _In_ const OrtHardwareDevice* device) {
  return OrtHardwareDeviceType(device->type);
}

ORT_API(const char*, HardwareDevice_Vendor, _In_ const OrtHardwareDevice* device) {
  return device->vendor.c_str();
}

ORT_API(const OrtKeyValuePairs*, HardwareDevice_Properties, _In_ const OrtHardwareDevice* device) {
  return &device->properties;
}

ORT_API(const char*, ExecutionDevice_EpName, _In_ const OrtExecutionDevice* exec_device) {
  return exec_device->ep_name.c_str();
}

ORT_API(const char*, ExecutionDevice_EpVendor, _In_ const OrtExecutionDevice* exec_device) {
  return exec_device->ep_vendor.c_str();
}

ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpProperties, _In_ const OrtExecutionDevice* exec_device) {
  return &exec_device->properties;
}

ORT_API(const OrtHardwareDevice*, ExecutionDevice_Device, _In_ const OrtExecutionDevice* exec_device) {
  return exec_device->device;
}

ORT_API_STATUS_IMPL(CreateExecutionDevice,
                    _In_ /*const*/ OrtEpApi::OrtEpFactory* ep_factory,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_reads_(num_ep_device_properties) const char** ep_device_properties_keys,
                    _In_reads_(num_ep_device_properties) const char** ep_device_properties_values,
                    _In_ size_t num_ep_device_properties,
                    _Out_ OrtExecutionDevice** ort_execution_device) {
  API_IMPL_BEGIN
  auto ed = std::make_unique<OrtExecutionDevice>();
  ed->ep_name = ep_factory->GetName(ep_factory);
  ed->ep_vendor = ep_factory->GetVendor(ep_factory);
  ed->device = hardware_device;
  for (size_t i = 0; i < num_ep_device_properties; ++i) {
    ed->properties.Add(ep_device_properties_keys[i], ep_device_properties_values[i]);
  }

  ed->ep_factory = ep_factory;

  *ort_execution_device = ed.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseExecutionDevice, _Frees_ptr_opt_ OrtExecutionDevice* device) {
  delete device;
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

    &OrtExecutionProviderApi::RegisterExecutionProviderLibrary,
    &OrtExecutionProviderApi::UnregisterExecutionProviderLibrary,

    &OrtExecutionProviderApi::HardwareDevice_Type,
    &OrtExecutionProviderApi::HardwareDevice_Vendor,
    &OrtExecutionProviderApi::HardwareDevice_Properties,
    &OrtExecutionProviderApi::ExecutionDevice_EpName,
    &OrtExecutionProviderApi::ExecutionDevice_EpVendor,
    &OrtExecutionProviderApi::ExecutionDevice_EpProperties,
    &OrtExecutionProviderApi::ExecutionDevice_Device,

    &OrtExecutionProviderApi::CreateExecutionDevice,
    &OrtExecutionProviderApi::ReleaseExecutionDevice,

    &OrtExecutionProviderApi::SessionOptionsConfigOptions,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, SessionOptionsConfigOption) / sizeof(void*) == 12,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}

}  // namespace OrtExecutionProviderApi

namespace onnxruntime {

Status EpLibraryPlugin::Load() {
  std::lock_guard<std::mutex> lock{mutex_};
  try {
    if (!factory_) {
      ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(library_path_, false, &handle_));

      OrtEpApi::GetEpFactoryFn factory_fn;
      ORT_RETURN_IF_ERROR(
          Env::Default().GetSymbolFromLibrary(handle_, "GetEpFactory", reinterpret_cast<void**>(&factory_fn)));

      factory_ = factory_fn(ep_name_.c_str(), OrtGetApiBase());
      ORT_RETURN_IF_ERROR(ToStatus(factory_->Initialize()));
    }
  } catch (const std::exception& ex) {
    // TODO: Add logging of exception
    auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load execution provider library: ", library_path_,
                                  " with error: ", ex.what());
    auto unload_status = Unload();  // If anything fails we unload the library
    if (!unload_status.IsOK()) {
      // TODO log unload status if not ok
    }
  }

  return Status::OK();
}

Status EpLibraryPlugin::Unload() {
  if (handle_) {
    if (factory_) {
      OrtStatusPtr status = factory_->Shutdown();  // should we still unload?
      if (status) {
        return ToStatus(status);
      }
    }

    ORT_RETURN_IF_ERROR(Env::Default().UnloadDynamicLibrary(handle_));
  }

  library_path_ = nullptr;
  factory_ = nullptr;
  handle_ = nullptr;

  return Status::OK();
}
}  // namespace onnxruntime
