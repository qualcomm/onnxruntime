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

  Status RegisterPluginEpFactory(OrtEpApi::OrtEpFactory& ep_factory) {
    ep_factories_.push_back(&ep_factory);
    return Status::OK();
  }

  Status RegisterLegacyEpFactory(const std::filesystem::path& /*library_path*/) {
    // create entry to load provider bridge EP from path and be able to call IExecutionProviderFactory.
    // TODO: Can we pass in session options and the session logger so those are setup upfront so things are more
    // consistent with the plugin EP creation?
    // One challenge is there's a `struct Logger` in the provider API which isn't the same as the Logger in the
    // C API. Can the C API OrtLogger be used?

    return Status::OK();
  }

  std::vector<std::unique_ptr<IExecutionProvider>>& CreateExecutionProviders(const OrtSessionOptions& so,
                                                                             const InferenceSession& session);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);
  Status Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                    const OrtThreadingOptions* tp_options = nullptr,
                    bool create_global_thread_pools = false);

  // register EPs that are built into the ORT binary
  std::vector<std::unique_ptr<IExecutionProvider>> CreateInternalEps(const OrtSessionOptions& so);
  void CreateLegacyEps(const OrtSessionOptions& so, std::vector<std::unique_ptr<IExecutionProvider>>& eps);
  void CreatePluginEps(const OrtSessionOptions& so, const OrtLogger& session_logger,
                       std::vector<std::unique_ptr<IExecutionProvider>>& eps);

  // EPs explicitly added to SessionOptions
  std::vector<std::unique_ptr<IExecutionProvider>> CreateSessionOptionEps(const OrtSessionOptions& so);

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
  bool create_global_thread_pools_{false};
  std::vector<AllocatorPtr> shared_allocators_;

  std::vector<std::shared_ptr<IExecutionProviderFactory>> provider_bridge_ep_factories_;
  std::vector<OrtEpApi::OrtEpFactory*> ep_factories_;
};
}  // namespace onnxruntime
