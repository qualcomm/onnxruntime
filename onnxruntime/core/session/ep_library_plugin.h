// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <mutex>

#include "core/session/ep_library.h"
#include "core/session/ep_library_provider_bridge.h"

namespace onnxruntime {
struct EpLibraryPlugin : EpLibrary {
  EpLibraryPlugin(const std::string& registration_name, const ORTCHAR_T* library_path)
      : registration_name_{registration_name},
        library_path_{library_path} {
  }

  const char* RegistrationName() const override {
    return registration_name_.c_str();
  }

  Status Load() override;

  const std::vector<OrtEpFactory*>& GetFactories() override {
    return factories_;
  }

  Status Unload() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryPlugin);

  // provider bridge has the EP factory functions as well as supporting onnxruntime::Provider.
  // we start by loading here to get the library handle and verify the OrtEpApi entry points are found.
  // if the provider bridge entry point is also found we pass the EpLibraryPlugin to an EpLibraryProviderBridge
  // that can which wraps the plugin and can also directly create an IExecutionProvider instance.
  static Status LoadPluginOrProviderBridge(const std::string& registration_name,
                                           const ORTCHAR_T* library_path,
                                           std::unique_ptr<EpLibrary>& ep_library,
                                           std::vector<EpFactoryInternal*>& internal_factories);

 private:
  std::mutex mutex_;
  const std::string registration_name_;
  const std::filesystem::path library_path_;
  void* handle_{};
  std::vector<OrtEpFactory*> factories_{};
  CreateEpApiFactoriesFn create_fn_{nullptr};
  ReleaseEpApiFactoryFn release_fn_{nullptr};
};
}  // namespace onnxruntime
