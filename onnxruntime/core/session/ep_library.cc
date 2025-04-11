// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/providers/cpu/cpu_execution_provider.h"
//
// #include "core/session/abi_devices.h"
// #include "core/session/abi_key_value_pairs.h"
// #include "core/session/abi_session_options_impl.h"
// #include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/environment.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

#if defined(USE_WEBGPU)
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#endif

namespace onnxruntime {

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
            LOGS_DEFAULT(ERROR) << "ReleaseEpFactory failed for: " << library_path_ << " with error: "
                                << ToStatus(status).ErrorMessage();
          }

          factories_[idx] = nullptr;  // clear the pointer
        }

        factories_.clear();
      } catch (const std::exception& ex) {
        auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load execution provider library: ", library_path_,
                                      " with error: ", ex.what());
      }
    }

    ORT_RETURN_IF_ERROR(Env::Default().UnloadDynamicLibrary(handle_));
  }

  library_path_ = nullptr;
  factories_.clear();
  handle_ = nullptr;

  return Status::OK();
}

namespace {
std::unique_ptr<EpLibraryInternal> CreateCpuEp() {
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // we have nothing to add.

      // use OrtApis directly to allocate or `new OrtKeyValuePairs` is fine as well. can add values directly
      // OrtApis::CreateKeyValuePairs(ep_metadata);
      //*ep_options = new OrtKeyValuePairs();
      //(*ep_options)->Add("options", "value");

      return true;
    }

    return false;
  };

  const auto create_cpu_ep = [](const SessionOptions& so, const logging::Logger& session_logger) {
    CPUExecutionProviderInfo epi{so.enable_cpu_mem_arena};
    auto ep = std::make_unique<CPUExecutionProvider>(epi);
    ep->SetLogger(&session_logger);
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
      // -

      // use OrtApis directly to allocate or `new OrtKeyValuePairs` is fine as well. can add values directly
      // OrtApis::CreateKeyValuePairs(ep_metadata);
      //*ep_options = new OrtKeyValuePairs();
      //(*ep_options)->Add("options", "value");

      return true;
    }

    return false;
  };

  const auto create_dml_ep = [](const SessionOptions& so, const logging::Logger& session_logger) {
    auto dml_ep_factory = DMLProviderFactoryCreator::Create(so.config_options,
                                                            /*device_id*/ 0,
                                                            /* skip_software_device_check*/ false,
                                                            /* disable_metacommands*/ false);
    auto dml_ep = dml_ep_factory->CreateProvider();
    dml_ep->SetLogger(&session_logger);
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

  const auto create_webgpu_ep = [](const SessionOptions& so, const logging::Logger& session_logger) {
    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(so.config_options);
    auto webgpu_ep = webgpu_ep_factory->CreateProvider();
    webgpu_ep->SetLogger(&session_logger);
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
