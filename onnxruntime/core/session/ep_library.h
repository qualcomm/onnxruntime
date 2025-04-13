// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "core/common/common.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/provider_bridge_ep_factory.h"
#include "core/session/provider_bridge_library.h"

namespace onnxruntime {

struct EpLibrary {
  virtual const char* RegistrationName() const = 0;
  virtual Status Load() { return Status::OK(); }
  virtual const std::vector<OrtEpApi::OrtEpFactory*>& GetFactories() = 0;  // valid after Load()
  virtual Status Unload() { return Status::OK(); }
};

struct EpLibraryInternal : EpLibrary {
  EpLibraryInternal(std::unique_ptr<InternalEpFactory> factory)
      : factory_{std::move(factory)}, factory_ptrs_{factory_.get()} {
  }

  const char* RegistrationName() const override {
    return factory_->GetName();  // same as EP name for internally registered libraries
  }

  const std::vector<OrtEpApi::OrtEpFactory*>& GetFactories() override {
    return factory_ptrs_;
  }

  // there's only ever one currently
  InternalEpFactory& GetInternalFactory() {
    return *factory_;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryInternal);

 private:
  std::unique_ptr<InternalEpFactory> factory_;         // all internal EPs register a single factory currently
  std::vector<OrtEpApi::OrtEpFactory*> factory_ptrs_;  // for convenience
};

struct EpLibraryProviderBridge : EpLibrary {
  EpLibraryProviderBridge(const std::string& registration_name, const ORTCHAR_T* library_path)
      : registration_name_{registration_name},
        library_path_{library_path},
        provider_library_{library_path} {
  }

  const char* RegistrationName() const override {
    return registration_name_.c_str();
  }

  const std::vector<OrtEpApi::OrtEpFactory*>& GetFactories() override {
    return factory_ptrs_;
  }

  // Provider bridge EPs are 'internal' as they can provide an IExecutionProvider instance directly
  // there's only ever one currently
  const std::vector<InternalEpFactory*>& GetInternalFactories() {
    return internal_factory_ptrs_;
  }

  Status Load() override;
  Status Unload() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryProviderBridge);

 private:
  std::string registration_name_;
  std::filesystem::path library_path_;
  ProviderLibrary provider_library_;  // handles onnxruntime_providers_shared and the provider bridge EP library
  std::vector<std::unique_ptr<InternalEpFactory>> factories_;
  std::vector<OrtEpApi::OrtEpFactory*> factory_ptrs_;      // for convenience
  std::vector<InternalEpFactory*> internal_factory_ptrs_;  // for convenience
};

struct EpLibraryPlugin : EpLibrary {
  EpLibraryPlugin(const std::string& registration_name, const ORTCHAR_T* library_path)
      : registration_name_{registration_name},
        library_path_{library_path} {
  }

  const char* RegistrationName() const override {
    return registration_name_.c_str();
  }

  Status Load() override;

  const std::vector<OrtEpApi::OrtEpFactory*>& GetFactories() override {
    return factories_;
  }

  Status Unload();

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryPlugin);

 private:
  std::mutex mutex_;
  const std::string registration_name_;
  const std::filesystem::path library_path_;
  void* handle_{};
  std::vector<OrtEpApi::OrtEpFactory*> factories_{};
};

// helper to create EpLibrary instances for all the EPs that are internally included in this build
struct InternalEpLibraryCreator {
  static std::vector<std::unique_ptr<EpLibraryInternal>> CreateInternalEps();
};

}  // namespace onnxruntime
