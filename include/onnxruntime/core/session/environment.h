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

#include "core/session/abi_devices.h"
#include "core/session/ep_library.h"
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
  Status CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo& mem_info,
                                      const std::unordered_map<std::string, std::string>& options,
                                      const OrtArenaCfg* arena_cfg = nullptr);

  Status RegisterExecutionProviderLibrary(const std::string& registration_name, const ORTCHAR_T* lib_path);
  Status UnregisterExecutionProviderLibrary(const std::string& registration_name);

  // convert an OrtEpFactory* to InternalEpFactory* if possible.
  InternalEpFactory* GetInternalEpFactory(OrtEpApi::OrtEpFactory* factory) const {
    // we're comparing pointers so the reinterpret_cast should be safe
    auto it = internal_ep_factories_.find(reinterpret_cast<InternalEpFactory*>(factory));
    return it != internal_ep_factories_.end() ? *it : nullptr;
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
  Status CreateAndRegisterInternalEps();

  Status RegisterExecutionProviderLibrary(const std::string& registration_name,
                                          std::unique_ptr<EpLibrary> ep_library,
                                          InternalEpFactory* internal_factory = nullptr);

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
  bool create_global_thread_pools_{false};
  std::vector<AllocatorPtr> shared_allocators_;

  struct EpInfo {
    // calls EpLibrary::Load
    // for each factory gets the OrtExecutionDevice instances and adds to execution_devices
    // internal_factory is set if this is an internal EP
    static Status Create(std::unique_ptr<EpLibrary> library_in, std::unique_ptr<EpInfo>& out,
                         InternalEpFactory* internal_factory = nullptr);

    // OrtEpApi::OrtEpFactory& GetFactory() {
    //   // Load must have been successful to get to here so this will should always be valid
    //   return *library->GetFactory();
    // }

    // removes entries for this library from execution_devices
    // calls EpLibrary::Unload
    ~EpInfo();

    std::unique_ptr<EpLibrary> library;
    std::vector<std::unique_ptr<OrtExecutionDevice>> execution_devices;
    InternalEpFactory* internal_factory{nullptr};  // used to populate/cleanup internal_ep_factories_
   private:
    EpInfo() = default;
  };

  // registration name to EpInfo for library
  std::unordered_map<std::string, std::unique_ptr<EpInfo>> ep_libraries_;

  // combined set of OrtExecutionDevices for all registered OrtEpFactory instances
  std::unordered_set<const OrtExecutionDevice*> execution_devices_;

  // lookup set for internal EPs so we can create an IExecutionProvider directly
  std::unordered_set<InternalEpFactory*> internal_ep_factories_;
};

}  // namespace onnxruntime
