// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_factory_internal.h"

#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_api_utils.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

using Forward = ForwardToFactory<EpFactoryInternal>;

EpFactoryInternal::EpFactoryInternal(const std::string& ep_name, const std::string& vendor,
                                     IsSupportedFunc&& is_supported_func,
                                     CreateFunc&& create_func)
    : ep_name_{ep_name},
      vendor_{vendor},
      is_supported_func_{std::move(is_supported_func)},
      create_func_{create_func} {
  // Constructor implementation
  ort_version_supported = ORT_API_VERSION;

  OrtEpFactory::GetName = Forward::GetFactoryName;
  OrtEpFactory::GetVendor = Forward::GetVendor;
  OrtEpFactory::GetDeviceInfoIfSupported = Forward::GetDeviceInfoIfSupported;
  OrtEpFactory::CreateEp = Forward::CreateEp;
  OrtEpFactory::ReleaseEp = Forward::ReleaseEp;
}

bool EpFactoryInternal::GetDeviceInfoIfSupported(const OrtHardwareDevice* device,
                                                 OrtKeyValuePairs** ep_device_metadata,
                                                 OrtKeyValuePairs** ep_options_for_device) const {
  return is_supported_func_(device, ep_device_metadata, ep_options_for_device);
}

OrtStatus* EpFactoryInternal::CreateEp(const OrtHardwareDevice* const* /*devices*/,
                                       const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                       size_t /*num_devices*/,
                                       const OrtSessionOptions* /*api_session_options*/,
                                       const OrtLogger* /*api_logger*/,
                                       OrtEpApi::OrtEp** /*ep*/) {
  ORT_THROW("Internal error. CreateIExecutionProvider should be used for EpFactoryInternal.");
}

OrtStatus* EpFactoryInternal::CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                                       const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                                       size_t num_devices,
                                                       const OrtSessionOptions* session_options,
                                                       const OrtLogger* session_logger,
                                                       std::unique_ptr<IExecutionProvider>& ep) {
  if (num_devices != 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpFactoryInternal only supports one device at a time.");
  }
  ep = create_func_(*session_options, *session_logger);

  return nullptr;
}

void EpFactoryInternal::ReleaseEp(OrtEpApi::OrtEp* /*ep*/) {
  // we never create an OrtEp so we should never be trying to release one
  ORT_THROW("Internal error. No ReleaseEp call is required for EpFactoryInternal.");
}
}  // namespace onnxruntime
