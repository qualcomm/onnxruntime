// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
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
#include "core/session/provider_bridge_ep_factory.h"
#include "core/session/ort_env.h"

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

#if defined(USE_WEBGPU)
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#endif

namespace onnxruntime {
namespace {
std::unique_ptr<ProviderBridgeEpFactory> CreateCudaEpFactory(Provider& provider) {
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
        device->vendor_id == 0x10de) {
      return true;
    }

    return false;
  };

  const auto create_cuda_ep = [&provider](const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
    OrtCUDAProviderOptionsV2 options;
    provider.UpdateProviderOptions(&options, session_options.value.config_options.configurations);
    auto ep_factory = provider.CreateExecutionProviderFactory(&options);
    auto ep = ep_factory->CreateProvider(session_options, session_logger);
    return ep;
  };

  std::string ep_name = kCudaExecutionProvider;
  auto factory = std::make_unique<ProviderBridgeEpFactory>(ep_name, "Microsoft", is_supported, create_cuda_ep);
  return factory;
}
}  // namespace

Status EpLibraryProviderBridge::Load() {
  auto& provider = provider_library_.Get();
  // Ideally the selection and creation funcs would come from Provider.
  // Start with local hardcoding for now. The set of EPs to support is constrained and it's an interim approach.
  // The matching on filename is fragile, but similar to what we do to load provider bridge EPs anyway.
  // Use the same library name to make it somewhat less fragile.
  // See https://github.com/microsoft/onnxruntime/blob/90c263f471bbce724e77d8e62831d3a9fa838b2f/onnxruntime/core/session/provider_bridge_ort.cc#L1782-L1815
  if (library_path_.filename().string().find("onnxruntime_providers_cuda") != std::string::npos) {
    factories_.push_back(CreateCudaEpFactory(provider));
  } else {
    ORT_NOT_IMPLEMENTED(
        "Execution provider library is not supported: ", library_path_);
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

      OrtEpApi::CreateEpFactoryFn create_fn;
      ORT_RETURN_IF_ERROR(
          Env::Default().GetSymbolFromLibrary(handle_, "CreateEpFactory", reinterpret_cast<void**>(&create_fn)));

      OrtEpApi::OrtEpFactory* factory;
      OrtStatus* status = create_fn(registration_name_.c_str(), OrtGetApiBase(), &factory);
      if (status != nullptr) {
        return ToStatus(status);
      }

      factories_.push_back(factory);
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

        for (size_t idx = 0, end = factories_.size(); idx < end; ++end) {
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
      // use OrtApis directly to allocate or `new OrtKeyValuePairs` is fine as well. can add values directly
      // OrtApis::CreateKeyValuePairs(ep_metadata);
      //*ep_options = new OrtKeyValuePairs();
      //(*ep_options)->Add("options", "value");

      return true;
    }

    return false;
  };

  const auto create_cpu_ep = [](const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
    CPUExecutionProviderInfo epi{session_options.value.enable_cpu_mem_arena};
    auto ep = std::make_unique<CPUExecutionProvider>(epi);
    ep->SetLogger(session_logger.ToInternal());
    return ep;
  };

  std::string ep_name = kCpuExecutionProvider;
  auto cpu_factory = std::make_unique<InternalEpFactory>(ep_name, "Microsoft", is_supported, create_cpu_ep);
  return std::make_unique<EpLibraryInternal>(std::move(cpu_factory));
}

#if defined(USE_DML)
std::unique_ptr<EpLibraryInternal> CreateDmlEp() {
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

  const auto create_dml_ep = [](const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
    auto dml_ep_factory = DMLProviderFactoryCreator::Create(session_options.value.config_options,
                                                            /*device_id*/ 0,
                                                            /* skip_software_device_check*/ false,
                                                            /* disable_metacommands*/ false);
    auto dml_ep = dml_ep_factory->CreateProvider();
    dml_ep->SetLogger(session_logger.ToInternal());
    return dml_ep;
  };

  std::string ep_name = kDmlExecutionProvider;
  auto dml_factory = std::make_unique<InternalEpFactory>(ep_name, "Microsoft", is_supported, create_dml_ep);

  return std::make_unique<EpLibraryInternal>(std::move(dml_factory));
}
#endif

#if defined(USE_WEBGPU)
std::unique_ptr<EpLibraryInternal> CreateWebGpuEp() {
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // does anything need to be added here?
      return true;
    }

    return false;
  };

  const auto create_webgpu_ep = [](const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(session_options.value.config_options);
    auto webgpu_ep = webgpu_ep_factory->CreateProvider();
    webgpu_ep->SetLogger(session_logger.ToInternal());
    return webgpu_ep;
  };

  std::string ep_name = kWebGpuExecutionProvider;
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
