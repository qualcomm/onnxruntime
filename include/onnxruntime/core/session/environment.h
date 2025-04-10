// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>

#include "core/common/common.h"
#include "core/common/basic_types.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"
#include "core/platform/device_discovery.h"
#include "core/platform/threadpool.h"

// TODO: should we be getting the plugin EP types this way or from an internal header?
#include "core/session/abi_devices.h"
#include "core/session/ep_api.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtThreadingOptions;
namespace onnxruntime {
class InferenceSession;
struct IExecutionProviderFactory;
struct SessionOptions;

/**
   Provides the runtime environment for onnxruntime.
   Create one instance for the duration of execution.
*/
class Environment {
 public:
  /**
     Create and initialize the runtime environment.
    @param logging manager instance that will enable per session logger output using
    session_options.session_logid as the logger id in messages.
    If nullptr, the default LoggingManager MUST have been created previously as it will be used
    for logging. This will use the default logger id in messages.
    See core/common/logging/logging.h for details, and how LoggingManager::DefaultLogger works.
    @param tp_options optional set of parameters controlling the number of intra and inter op threads for the global
    threadpools.
    @param create_global_thread_pools determine if this function will create the global threadpools or not.
  */
  static Status Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                       std::unique_ptr<Environment>& environment,
                       const OrtThreadingOptions* tp_options = nullptr,
                       bool create_global_thread_pools = false);

  logging::LoggingManager* GetLoggingManager() const {
    return logging_manager_.get();
  }

  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager) {
    logging_manager_ = std::move(logging_manager);
  }

  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPool() const {
    return intra_op_thread_pool_.get();
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPool() const {
    return inter_op_thread_pool_.get();
  }

  bool EnvCreatedWithGlobalThreadPools() const {
    return create_global_thread_pools_;
  }

  /**
   * Registers an allocator for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   */
  Status RegisterAllocator(AllocatorPtr allocator);

  /**
   * Creates and registers an allocator for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   */
  Status CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg = nullptr);

  /**
   * Returns the list of registered allocators in this env.
   */
  const std::vector<AllocatorPtr>& GetRegisteredSharedAllocators() const {
    return shared_allocators_;
  }

  /**
   * Removes registered allocator that was previously registered for sharing between multiple sessions.
   */
  Status UnregisterAllocator(const OrtMemoryInfo& mem_info);

  Environment() = default;

  /**
   * Create and register an allocator, specified by provider_type, for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   * For provider_type please refer core/graph/constants.h
   */
  Status CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo& mem_info, const std::unordered_map<std::string, std::string>& options, const OrtArenaCfg* arena_cfg = nullptr);

  // auto EP selection
  // possibly belongs in separate class.
  std::vector<std::unique_ptr<IExecutionProvider>> CreateExecutionProviders(const OrtSessionOptions& so,
                                                                            const InferenceSession& session);

  Status RegisterExecutionProviderLibrary(const ORTCHAR_T* lib_path, const std::string& ep_name);
  Status UnregisterExecutionProviderLibrary(const std::string& ep_name);

  void RegisterInternalExecutionProvider(const OrtEpApi::OrtEp& ep, const IExecutionProvider& internal_ep) {
    ortep_to_internal_ep_[&ep] = &internal_ep;
  }

  const IExecutionProvider* GetInternalExecutionProvider(const OrtEpApi::OrtEp& ep) const {
    auto it = ortep_to_internal_ep_.find(&ep);
    if (it != ortep_to_internal_ep_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void UnregisterInternalExecutionProvider(const OrtEpApi::OrtEp& ep) {
    ortep_to_internal_ep_.erase(&ep);
  }

  const std::unordered_set<const OrtExecutionDevice*> GetExecutionDevices() const {
    return execution_devices_;
  }

  ~Environment();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);

  Status Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                    const OrtThreadingOptions* tp_options = nullptr,
                    bool create_global_thread_pools = false);

  // register EPs that are built into the ORT binary so they can take part in AutoEP selection
  // added to ep_libraries
  Status CreateInternalEps();

  Status RegisterExecutionProviderLibrary(std::unique_ptr<EpLibrary> ep_library, const std::string& ep_name);

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
  bool create_global_thread_pools_{false};
  std::vector<AllocatorPtr> shared_allocators_;

  struct EpInfo {
    static Status Create(std::unique_ptr<EpLibrary> library_in, std::unique_ptr<EpInfo>& out) {
      if (!library_in) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "EpLibrary was null");
      }

      out.reset(new EpInfo());  // can't use make_unique with private ctor
      EpInfo& instance = *out;
      instance.library = std::move(library_in);

      ORT_RETURN_IF_ERROR(instance.library->Load());
      auto& factory = instance.GetFactory();

      // for each device
      for (const auto& device : DeviceDiscovery::GetDevices()) {
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;

        if (factory.GetDeviceInfoIfSupported(&factory, &device, &ep_metadata, &ep_options)) {
          auto ed = std::make_unique<OrtExecutionDevice>();
          ed->ep_name = factory.GetName(&factory);  // creating the OrtExecutionDevice here means the EP name is always fixed
          ed->ep_vendor = factory.GetVendor(&factory);
          ed->device = &device;
          if (ep_metadata) {
            ed->ep_metadata = *ep_metadata;
          }
          if (ep_options) {
            ed->ep_options = *ep_options;
          }

          ed->ep_factory = &factory;

          instance.execution_devices.push_back(std::move(ed));
        }
      }

      return Status::OK();
    }

    OrtEpApi::OrtEpFactory& GetFactory() {
      // Load must have been successful to get to here to this will should always be valid
      return *library->GetFactory();
    }

    ~EpInfo() {
      execution_devices.clear();
      auto status = library->Unload();
      if (!status.IsOK()) {
        LOGS_DEFAULT(WARNING) << "Failed to unload EP library: " << library->Name() << " with error: " << status.ErrorMessage();
      }
    }

    std::unique_ptr<EpLibrary> library;
    std::vector<std::unique_ptr<OrtExecutionDevice>> execution_devices;

   private:
    EpInfo() = default;
  };

  std::unordered_map<std::string, std::unique_ptr<EpInfo>> ep_libraries_;  // EP name to things it provides
  std::unordered_set<const OrtExecutionDevice*> execution_devices_;

  // map for internal EPs so we can use IExecutionProvider for them instead of OrtEpApi::OrtEp
  std::unordered_map<const OrtEpApi::OrtEp*, const IExecutionProvider*> ortep_to_internal_ep_;
};

}  // namespace onnxruntime
