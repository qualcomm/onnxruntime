// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/provider_options.h"
#include "core/framework/session_options.h"
#include "core/providers/providers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/shared_library/provider_host_api.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_env.h"

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

#if defined(USE_WEBGPU)
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#endif

namespace onnxruntime {
namespace {
// many EPs parse the options from prior to them being added to session options.
// to support that we need to extract the EP specific options from session_options and remove the prefix.
ProviderOptions GetOptionsFromSessionOptions(const std::string& ep_name, const SessionOptions& session_options) {
  const std::string option_prefix = ProviderOptionsUtils::GetProviderOptionPrefix(ep_name);
  ProviderOptions ep_options;

  for (const auto& [key, value] : session_options.config_options.configurations) {
    if (key.find(option_prefix) == 0) {
      // remove the prefix and add
      ep_options[key.substr(option_prefix.length())] = value;
    }
  }

  return ep_options;
}

std::unique_ptr<InternalEpFactory> CreateCudaEpFactory(Provider& provider) {
  // Use the name that SessionOptionsAppendExecutionProvider uses to identify the EP as that matches the
  // expected name in the configuration options. must be static to be valid for the lambdas
  static const std::string ep_name = "CUDA";

  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
        device->vendor_id == 0x10de) {
      return true;
    }

    return false;
  };

  const auto create_cuda_ep = [&provider](const OrtSessionOptions& session_options,
                                          const OrtLogger& session_logger) {
    OrtCUDAProviderOptionsV2 options;
    const SessionOptions& so = session_options.existing_value ? **session_options.existing_value
                                                              : session_options.value;

    auto ep_options = GetOptionsFromSessionOptions(ep_name, so);
    provider.UpdateProviderOptions(&options, ep_options);

    auto ep_factory = provider.CreateExecutionProviderFactory(&options);
    auto ep = ep_factory->CreateProvider(session_options, session_logger);

    return ep;
  };

  auto factory = std::make_unique<InternalEpFactory>(ep_name, "Microsoft", is_supported, create_cuda_ep);
  return factory;
}
}  // namespace

Status EpLibraryProviderBridge::Load() {
  auto& provider = provider_library_.Get();
  // Ideally the selection and creation funcs would come from Provider.
  // Start with local hardcoding using the same library names as provider_bridge_ort.cc.
  // The set of EPs to support is constrained and it's a short term approach.
  // See https://github.com/microsoft/onnxruntime/blob/90c263f471bbce724e77d8e62831d3a9fa838b2f/onnxruntime/core/session/provider_bridge_ort.cc#L1782-L1815
  if (library_path_.filename().string().find("onnxruntime_providers_cuda") != std::string::npos) {
    auto ep_factory = CreateCudaEpFactory(provider);
    factory_ptrs_.push_back(ep_factory.get());
    internal_factory_ptrs_.push_back(ep_factory.get());
    factories_.push_back(std::move(ep_factory));
  } else {
    ORT_NOT_IMPLEMENTED("Execution provider library is not supported: ", library_path_);
  }

  return Status::OK();
}

Status EpLibraryProviderBridge::Unload() {
  provider_library_.Unload();
  return Status::OK();
}

Status EpLibraryPlugin::Load() {
  std::lock_guard<std::mutex> lock{mutex_};
  try {
    if (factories_.empty()) {
      ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(library_path_, false, &handle_));

      OrtEpApi::CreateEpFactoriesFn create_fn;
      ORT_RETURN_IF_ERROR(
          Env::Default().GetSymbolFromLibrary(handle_, "CreateEpFactories", reinterpret_cast<void**>(&create_fn)));

      // allocate buffer for EP to add factories to.
      std::vector<OrtEpApi::OrtEpFactory*> factories{4, nullptr};

      size_t num_factories = 0;
      OrtStatus* status = create_fn(registration_name_.c_str(), OrtGetApiBase(), factories.data(), factories.size(),
                                    &num_factories);
      if (status != nullptr) {
        return ToStatus(status);
      }

      for (size_t i = 0; i < num_factories; ++i) {
        factories_.push_back(factories[i]);
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
  // Call ReleaseEpFactory for all factories and unload the library.
  // Current implementation assume any error is permanent so does not leave pieces around to re-attempt Unload.
  if (handle_) {
    if (!factories_.empty()) {
      try {
        OrtEpApi::ReleaseEpFactoryFn release_fn;
        ORT_RETURN_IF_ERROR(
            Env::Default().GetSymbolFromLibrary(handle_, "ReleaseEpFactory", reinterpret_cast<void**>(&release_fn)));

        for (size_t idx = 0, end = factories_.size(); idx < end; ++idx) {
          auto* factory = factories_[idx];
          if (factory == nullptr) {
            continue;
          }

          OrtStatus* status = release_fn(factory);
          if (status != nullptr) {
            // log it and treat it as released
            LOGS_DEFAULT(ERROR) << "ReleaseEpFactory failed for: " << library_path_ << " with error: "
                                << ToStatus(status).ErrorMessage();
          }

          factories_[idx] = nullptr;  // clear the pointer in case there's a failure before all are released
        }

        factories_.clear();
      } catch (const std::exception& ex) {
        LOGS_DEFAULT(ERROR) << "Failed releasing EP factories from " << library_path_ << ": " << ex.what();
      }
    }

    // TODO: Is there a better way? Is it worth worrying about?
    if (!factories_.empty()) {
      LOGS_DEFAULT(ERROR) << "Unloading " << library_path_ << ". " << factories_.size()
                          << " factories were not released due to errors. This may cause memory leaks. "
                             "Please check the error details in the log.";
    }

    ORT_RETURN_IF_ERROR(Env::Default().UnloadDynamicLibrary(handle_));
  }

  handle_ = nullptr;

  return Status::OK();
}

namespace {
std::unique_ptr<EpLibraryInternal> CreateCpuEp() {
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      return true;
    }

    return false;
  };

  const auto create_cpu_ep = [](const OrtSessionOptions& session_options,
                                const OrtLogger& session_logger) {
    CPUExecutionProviderInfo epi{session_options.value.enable_cpu_mem_arena};
    auto ep = std::make_unique<CPUExecutionProvider>(epi);
    ep->SetLogger(session_logger.ToInternal());
    return ep;
  };

  std::string ep_name = "CPU";
  auto cpu_factory = std::make_unique<InternalEpFactory>(ep_name, "Microsoft", is_supported, create_cpu_ep);
  return std::make_unique<EpLibraryInternal>(std::move(cpu_factory));
}

#if defined(USE_DML)
std::unique_ptr<EpLibraryInternal> CreateDmlEp() {
  static const std::string ep_name = "DML";
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // does anything need to be added here?
      // - can we get the device id and set in ep_options?
      //   - is that 'Bus number' in windows? is that already in device->bus_id?
      //   - or should that be in ep_metadata so we can use in the call to DMLProviderFactoryCreator::Create?
      return true;
    }

    return false;
  };

  const auto create_dml_ep = [](const OrtSessionOptions& session_options,
                                const OrtLogger& session_logger) {
    const SessionOptions& so = session_options.existing_value ? **session_options.existing_value
                                                              : session_options.value;

    auto ep_options = GetOptionsFromSessionOptions(ep_name, so);
    auto dml_ep_factory = DMLProviderFactoryCreator::CreateFromProviderOptions(so.config_options,
                                                                               ep_options);

    auto dml_ep = dml_ep_factory->CreateProvider();
    dml_ep->SetLogger(session_logger.ToInternal());
    return dml_ep;
  };

  auto dml_factory = std::make_unique<InternalEpFactory>(ep_name, "Microsoft", is_supported, create_dml_ep);

  return std::make_unique<EpLibraryInternal>(std::move(dml_factory));
}
#endif

#if defined(USE_WEBGPU)
std::unique_ptr<EpLibraryInternal> CreateWebGpuEp() {
  static const std::string ep_name = "WebGPU";

  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // does anything need to be added here?
      return true;
    }

    return false;
  };

  const auto create_webgpu_ep = [](const OrtSessionOptions& session_options,
                                   const OrtLogger& session_logger) {
    const SessionOptions& so = session_options.existing_value ? **session_options.existing_value
                                                              : session_options.value;

    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(so.config_options);
    auto webgpu_ep = webgpu_ep_factory->CreateProvider();
    webgpu_ep->SetLogger(session_logger.ToInternal());
    return webgpu_ep;
  };

  auto webgpu_factory = std::make_unique<InternalEpFactory>(ep_name, "Microsoft", is_supported, create_webgpu_ep);

  return std::make_unique<EpLibraryInternal>(std::move(webgpu_factory));
}
#endif
}  // namespace

std::vector<std::unique_ptr<EpLibraryInternal>> InternalEpLibraryCreator::CreateInternalEps() {
  std::vector<std::unique_ptr<EpLibraryInternal>> internal_eps;
  internal_eps.reserve(4);

  // CPU EP
  internal_eps.push_back(CreateCpuEp());

#if defined(USE_DML)
  internal_eps.push_back(CreateDmlEp());
#endif

#if defined(USE_WEBGPU)
  internal_eps.push_back(CreateWebGpuEp());
#endif

  return internal_eps;
}

}  // namespace onnxruntime
