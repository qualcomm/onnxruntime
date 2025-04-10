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
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;
namespace OrtExecutionProviderApi {
ORT_API_STATUS_IMPL(RegisterExecutionProviderLibrary, _In_ OrtEnv* env, const ORTCHAR_T* path, const char* ep_name) {
  API_IMPL_BEGIN
  // provider bridge EPs can provide path to library to load in session options.
  // might need logic to convert from session option ConfigParams to the EP specific settings
  //
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().RegisterExecutionProviderLibrary(path, ep_name));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(UnregisterExecutionProviderLibrary, _In_ OrtEnv* env, const char* ep_name) {
  API_IMPL_BEGIN
  //
  ORT_API_RETURN_IF_STATUS_NOT_OK(env->GetEnvironment().UnregisterExecutionProviderLibrary(ep_name));
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

namespace onnxruntime {

Status EpLibraryPlugin::Load() {
  std::lock_guard<std::mutex> lock{mutex_};
  try {
    if (!factory_) {
      ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(library_path_, false, &handle_));

      OrtEpApi::CreateEpFactoryFn create_fn;
      ORT_RETURN_IF_ERROR(
          Env::Default().GetSymbolFromLibrary(handle_, "CreateEpFactory", reinterpret_cast<void**>(&create_fn)));

      OrtStatus* status = create_fn(ep_name_.c_str(), OrtGetApiBase(), &factory_);
      if (status != nullptr) {
        return ToStatus(status);
      }
    }
  } catch (const std::exception& ex) {
    // TODO: Add logging of exception
    auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load execution provider library: ", library_path_,
                                  " with error: ", ex.what());
    auto unload_status = Unload();  // If anything fails we unload the library
    if (!unload_status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Failed to unload execution provider library: " << library_path_ << " with error: "
                          << unload_status.ErrorMessage();
    }
  }

  return Status::OK();
}

Status EpLibraryPlugin::Unload() {
  if (handle_) {
    if (factory_) {
      try {
        OrtEpApi::ReleaseEpFactoryFn release_fn;
        ORT_RETURN_IF_ERROR(
            Env::Default().GetSymbolFromLibrary(handle_, "ReleaseEpFactory", reinterpret_cast<void**>(&release_fn)));

        OrtStatus* status = release_fn(factory_);
        if (status != nullptr) {
          LOGS_DEFAULT(ERROR) << "ReleaseEpFactory failed for: " << library_path_ << " with error: "
                              << ToStatus(status).ErrorMessage();
        }

      } catch (const std::exception& ex) {
        auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load execution provider library: ", library_path_,
                                      " with error: ", ex.what());
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
