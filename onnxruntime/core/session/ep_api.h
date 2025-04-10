// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <mutex>

#include "core/common/common.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"

namespace OrtExecutionProviderApi {

// implementation that returns the API struct
ORT_API(const OrtEpApi*, GetEpApi);

ORT_API_STATUS_IMPL(RegisterExecutionProviderLibrary, _In_ OrtEnv* env, const ORTCHAR_T* path, const char* ep_name);
ORT_API_STATUS_IMPL(UnregisterExecutionProviderLibrary, _In_ OrtEnv* env, _In_ const char* ep_name);

// OrtHardwareDevice and OrtExecutionDevice accessors
ORT_API(OrtHardwareDeviceType, HardwareDevice_Type, _In_ const OrtHardwareDevice* device);
ORT_API(int32_t, HardwareDevice_VendorId, _In_ const OrtHardwareDevice* device);
ORT_API(const char*, HardwareDevice_Vendor, _In_ const OrtHardwareDevice* device);
ORT_API(int32_t, HardwareDevice_BusId, _In_ const OrtHardwareDevice* device);
ORT_API(const OrtKeyValuePairs*, HardwareDevice_Properties, _In_ const OrtHardwareDevice* device);

// ORT_API(const char*, ExecutionDevice_EpName, _In_ const OrtExecutionDevice* device);
// ORT_API(const char*, ExecutionDevice_EpVendor, _In_ const OrtExecutionDevice* device);
// ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpMetadata, _In_ const OrtExecutionDevice* device);
// ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpOptions, _In_ const OrtExecutionDevice* device);
// ORT_API(const OrtHardwareDevice*, ExecutionDevice_Device, _In_ const OrtExecutionDevice* device);

// user must call ReleaseKeyValuePairs when done.
ORT_API_STATUS_IMPL(SessionOptionsConfigOptions, _In_ const OrtSessionOptions* session_options,
                    _Out_ OrtKeyValuePairs** options);

// Get ConfigOptions by key. Returns null in value if key not found (vs pointer to empty string if found).
ORT_API(const char*, SessionOptionsConfigOption, _In_ const OrtSessionOptions* session_options, _In_ const char* key);

}  // namespace OrtExecutionProviderApi

namespace onnxruntime {
struct EpLibrary {
  virtual const char* Name() const = 0;
  virtual Status Load() { return Status::OK(); }
  virtual OrtEpApi::OrtEpFactory* GetFactory() = 0;  // valid after Load()
  virtual Status Unload() { return Status::OK(); }
};

struct EpLibraryInternal : EpLibrary {
  EpLibraryInternal(std::unique_ptr<InternalEpFactory> factory)
      : factory_{std::move(factory)}, factory_ptr_{factory_.get()} {
  }

  const char* Name() const override {
    return factory_ptr_->GetName(factory_ptr_);
  }

  OrtEpApi::OrtEpFactory* GetFactory() override {
    return factory_ptr_;
  }

 private:
  std::unique_ptr<InternalEpFactory> factory_;
  OrtEpApi::OrtEpFactory* factory_ptr_;  // for convenience
};

struct EpLibraryProviderBridge : EpLibrary {
  // can we extract Provider from provider_bridge_ort.cc and plug it in here?
};

// this is based on Provider in provider_bridge_ort.cc
// TODO: is Stuart's way better?
struct EpLibraryPlugin : EpLibrary {
  EpLibraryPlugin(const std::string& ep_name, const ORTCHAR_T* library_path)
      : ep_name_{ep_name}, library_path_{library_path} {
  }

  const char* Name() const override {
    assert(factory_);
    return factory_->GetName(factory_);
  }

  Status Load() override;

  OrtEpApi::OrtEpFactory* GetFactory() override {
    return factory_;
  }

  Status Unload();

 private:
  std::mutex mutex_;
  const std::string ep_name_;
  const ORTCHAR_T* library_path_;
  OrtEpApi::OrtEpFactory* factory_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryPlugin);
};

}  // namespace onnxruntime
