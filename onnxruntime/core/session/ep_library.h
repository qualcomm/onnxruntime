// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "core/common/common.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class InternalEpFactory;

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
    return factory_->GetName();  // internally registered so same as ep name
  }

  const std::vector<OrtEpApi::OrtEpFactory*>& GetFactories() override {
    return factory_ptrs_;
  }

  // there's only ever one currently
  InternalEpFactory& GetInternalFactory() {
    return *factory_;
  }

 private:
  std::unique_ptr<InternalEpFactory> factory_;
  std::vector<OrtEpApi::OrtEpFactory*> factory_ptrs_;  // for convenience
};

struct EpLibraryProviderBridge : EpLibrary {
  // can we extract Provider from provider_bridge_ort.cc and plug it in here?
};

// this is based on Provider in provider_bridge_ort.cc
// TODO: is Stuart's way better?
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

 private:
  std::mutex mutex_;
  const std::string registration_name_;
  const ORTCHAR_T* library_path_;
  std::vector<OrtEpApi::OrtEpFactory*> factories_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryPlugin);
};

// helper to create EpLibrary instances for all the EPs that are internally included in this build
struct InternalEpLibraryCreator {
  static std::vector<std::unique_ptr<EpLibraryInternal>> CreateInternalEps();
};

}  // namespace onnxruntime
