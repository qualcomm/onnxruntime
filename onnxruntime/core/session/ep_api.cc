// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/provider_options.h"
#include "core/providers/providers.h"
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

ORT_API_STATUS_IMPL(GetEpDevices, _In_ const OrtEnv* env,
                    _Outptr_ const OrtEpDevice* const** ep_devices, _Out_ size_t* num_ep_devices) {
  API_IMPL_BEGIN
  const auto& execution_devices = env->GetEnvironment().GetOrtEpDevices();
  *ep_devices = execution_devices.data();
  *num_ep_devices = execution_devices.size();

  return nullptr;
  API_IMPL_END
}

struct ExecutionProviderFactory : public IExecutionProviderFactory {
 public:
  ExecutionProviderFactory(EpFactoryInternal& ep_factory, std::vector<const OrtEpDevice*> ep_devices)
      : ep_factory_{ep_factory} {
    devices_.reserve(ep_devices.size());
    ep_metadata_.reserve(ep_devices.size());

    for (const auto* ep_device : ep_devices) {
      devices_.push_back(ep_device->device);
      ep_metadata_.push_back(&ep_device->ep_metadata);
    }
  }

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    std::unique_ptr<IExecutionProvider> ep;
    OrtStatus* status = ep_factory_.CreateIExecutionProvider(devices_.data(), ep_metadata_.data(), devices_.size(),
                                                             &session_options, &session_logger, ep);
    if (status != nullptr) {
      ORT_THROW("Error creating execution provider: ", ToStatus(status).ToString());
    }

    return ep;
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ORT_NOT_IMPLEMENTED("CreateProvider without parameters is not supported.");
  }

 private:
  EpFactoryInternal& ep_factory_;
  std::vector<const OrtHardwareDevice*> devices_;
  std::vector<const OrtKeyValuePairs*> ep_metadata_;
};

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_V2, _In_ OrtSessionOptions* session_options,
                    _In_ OrtEnv* env, _In_ const char* ep_name_in,
                    _In_reads_(num_op_options) const char* const* ep_option_keys,
                    _In_reads_(num_op_options) const char* const* ep_option_vals,
                    size_t num_ep_options) {
  API_IMPL_BEGIN
  const auto& execution_devices = env->GetEnvironment().GetOrtEpDevices();
  std::string ep_name{ep_name_in};
  std::vector<const OrtEpDevice*> ep_devices;

  EpFactoryInternal* internal_factory = nullptr;
  for (const auto& entry : execution_devices) {
    if (entry->ep_name == ep_name) {
      internal_factory = env->GetEnvironment().GetEpFactoryInternal(entry->ep_factory);
      if (!internal_factory) {
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "EP is not currently supported by this API");
      }

      ep_devices.push_back(entry);

      // add the options to the session options with the EP prefix.
      // first add the default values with prefix followed by user specified values so those win
      const auto prefix = ProviderOptionsUtils::GetProviderOptionPrefix(entry->ep_name);
      auto& config_options = session_options->value.config_options;
      for (const auto& [key, value] : entry->ep_options.entries) {
        ORT_API_RETURN_IF_STATUS_NOT_OK(config_options.AddConfigEntry((prefix + key).c_str(), value.c_str()));
      }

      for (size_t i = 0; i < num_ep_options; ++i) {
        if (ep_option_keys[i] == nullptr) {
          continue;
        }

        ORT_API_RETURN_IF_STATUS_NOT_OK(config_options.AddConfigEntry((prefix + ep_option_keys[i]).c_str(),
                                                                      ep_option_vals[i]));
      }
    }
  }

  if (internal_factory) {
    session_options->provider_factories.push_back(
        std::make_unique<ExecutionProviderFactory>(*internal_factory, ep_devices));
  }

  return nullptr;
  API_IMPL_END
}

// OrtEpDevice accessors
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

ORT_API(const OrtKeyValuePairs*, HardwareDevice_Metadata, _In_ const OrtHardwareDevice* ep_device) {
  return &ep_device->properties;
}

ORT_API(const char*, EpDevice_EpName, _In_ const OrtEpDevice* ep_device) {
  return ep_device->ep_name.c_str();
}

ORT_API(const char*, EpDevice_EpVendor, _In_ const OrtEpDevice* ep_device) {
  return ep_device->ep_vendor.c_str();
}

ORT_API(const OrtKeyValuePairs*, EpDevice_EpMetadata, _In_ const OrtEpDevice* ep_device) {
  return &ep_device->ep_metadata;
}

ORT_API(const OrtKeyValuePairs*, EpDevice_EpOptions, _In_ const OrtEpDevice* ep_device) {
  return &ep_device->ep_options;
}

ORT_API(const OrtHardwareDevice*, EpDevice_Device, _In_ const OrtEpDevice* ep_device) {
  return ep_device->device;
}

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
ORT_API(const char*, SessionOptions_GetConfigOption, _In_ const OrtSessionOptions* session_options, _In_ const char* key) {
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

    // ??? Location???
    // Should these and all the OrtHardwareDevice and OrtEpDevice related things be in the base ORT API?
    // A general ORT user might register/unregister a library, discover the OrtEpDevices and use
    // SessionOptionsAppendExecutionProvider_V2 without caring about all the other pieces which are used by an
    // EP library author.
    &OrtExecutionProviderApi::RegisterExecutionProviderLibrary,
    &OrtExecutionProviderApi::UnregisterExecutionProviderLibrary,
    &OrtExecutionProviderApi::GetEpDevices,
    &OrtExecutionProviderApi::SessionOptionsAppendExecutionProvider_V2,

    &OrtExecutionProviderApi::HardwareDevice_Type,
    &OrtExecutionProviderApi::HardwareDevice_VendorId,
    &OrtExecutionProviderApi::HardwareDevice_Vendor,
    &OrtExecutionProviderApi::HardwareDevice_BusId,
    &OrtExecutionProviderApi::HardwareDevice_Metadata,

    &OrtExecutionProviderApi::EpDevice_EpName,
    &OrtExecutionProviderApi::EpDevice_EpVendor,
    &OrtExecutionProviderApi::EpDevice_EpMetadata,
    &OrtExecutionProviderApi::EpDevice_EpOptions,
    &OrtExecutionProviderApi::EpDevice_Device,

    &OrtExecutionProviderApi::SessionOptions_GetConfigOptions,
    &OrtExecutionProviderApi::SessionOptions_GetConfigOption,
    &OrtExecutionProviderApi::SessionOptions_GetOptimizationLevel,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, SessionOptions_GetOptimizationLevel) / sizeof(void*) == 16,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}

}  // namespace OrtExecutionProviderApi
