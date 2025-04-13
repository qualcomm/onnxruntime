// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include "core/common/common.h"
// #include "core/framework/execution_provider.h"
// #include "core/session/onnxruntime_c_api.h"
//
// namespace onnxruntime {
// struct EpLibraryProviderBridge;
// struct SessionOptions;
//
// class ProviderBridgeEpFactory : public OrtEpApi::OrtEpFactory {
//  public:
//   using IsSupportedFunc = std::function<bool(const OrtHardwareDevice* device,
//                                              OrtKeyValuePairs** ep_metadata,
//                                              OrtKeyValuePairs** ep_options)>;
//
//   using CreateFunc = std::function<std::unique_ptr<IExecutionProvider>(const OrtSessionOptions& session_options,
//                                                                        const OrtLogger& logger)>;
//   ProviderBridgeEpFactory(const std::string& ep_name, const std::string& vendor,
//                           IsSupportedFunc&& is_supported_func,
//                           CreateFunc&& create_func);
//
//   const char* GetName() const { return ep_name_.c_str(); }
//   const char* GetVendor() const { return vendor_.c_str(); }
//
//   bool GetDeviceInfoIfSupported(_In_ const OrtHardwareDevice* device,
//                                 _Out_ OrtKeyValuePairs** ep_device_metadata,
//                                 _Out_ OrtKeyValuePairs** ep_options_for_device) const;
//
//   // we don't implement this. code should check
//   OrtStatus* CreateEp(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
//                       _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata_pairs,
//                       _In_ size_t num_devices,
//                       _In_ const OrtSessionOptions* session_options,
//                       _In_ const OrtLogger* logger, _Out_ OrtEpApi::OrtEp** ep);
//
//   // we implement this. provide the same args in case we need something from device or ep_metadata_pairs in the future.
//   // TODO: reutrn unique_ptr or shared_ptr? latter might be more flexible as it doesn't require the implementer to
//   // create a new instance every time. it's going to become a shared_ptr when added to InferenceSession anyway.
//   OrtStatus* CreateIExecutionProvider(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
//                                       _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata_pairs,
//                                       _In_ size_t num_devices,
//                                       _In_ const OrtSessionOptions* session_options,
//                                       _In_ const OrtLogger* logger, _Out_ std::shared_ptr<IExecutionProvider>& ep);
//
//   // Function ORT calls to release an EP instance.
//   void ReleaseEp(OrtEpApi::OrtEp* ep);
//
//  private:
//   const std::string ep_name_;                // EP name library was registered with
//   const std::string vendor_;                 // EP vendor name
//   const IsSupportedFunc is_supported_func_;  // function to check if the device is supported
//   const CreateFunc create_func_;             // function to create the EP instance
//
//   std::vector<std::unique_ptr<ProviderBridgeEpFactory>> eps_;  // EP instances created by this factory
// };
//
// }  // namespace onnxruntime
