// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"
#include "core/platform/device_discovery.h"
#include "core/platform/threadpool.h"

// TODO: should we be getting the plugin EP types this way or from an internal header?
#include "core/session/onnxruntime_c_api.h"

struct OrtThreadingOptions;
namespace onnxruntime {
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

  Status RegisterProviderBridgeEpFactory(const std::filesystem::path& library_path);

  Status RegisterPluginEP(OrtEpApi::OrtEpPlugin& ep) {
    plugin_eps_.push_back(&ep);
    OrtEpApi::OrtEpFactory* ep_factory = nullptr;

    // TODO: Weird to be calling API funcs here as we need to convert from OrtStatus -> Status -> OrtStatus.
    //       Can we structure this differently? Do we need OrtEpPlugin and OrtEpFactory or could we just register
    //       the factory here?
    OrtStatus* status = ep.CreateEpFactory(&ep, &ep_factory);
    if (status != nullptr) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Failed to create EP factory. <FIXME: add error>");
    }

    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);
  Status Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                    const OrtThreadingOptions* tp_options = nullptr,
                    bool create_global_thread_pools = false);

  Status RegisterInternalEpFactories();

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
  bool create_global_thread_pools_{false};
  std::vector<AllocatorPtr> shared_allocators_;

  std::vector<OrtEpApi::OrtEpPlugin*> plugin_eps_;
  std::vector<OrtEpApi::OrtEpFactory*> plugin_ep_factories_;
};
}  // namespace onnxruntime
