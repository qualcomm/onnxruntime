// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <mutex>

#include "core/common/common.h"
#include "core/session/ep_factory_internal.h"
#include "core/session/onnxruntime_c_api.h"

namespace OrtExecutionProviderApi {

// implementation that returns the API struct
ORT_API(const OrtEpApi*, GetEpApi);

ORT_API_STATUS_IMPL(RegisterExecutionProviderLibrary, _In_ OrtEnv* env, const char* ep_name, const ORTCHAR_T* path);
ORT_API_STATUS_IMPL(UnregisterExecutionProviderLibrary, _In_ OrtEnv* env, _In_ const char* ep_name);

ORT_API_STATUS_IMPL(GetEpDevices, _In_ const OrtEnv* env,
                    _Outptr_ const OrtEpDevice* const** ep_devices, _Out_ size_t* num_ep_devices);

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_V2, _In_ OrtSessionOptions* sess_options,
                    _In_ OrtEnv* env, _In_ const char* ep_name,
                    _In_reads_(num_op_options) const char* const* ep_option_keys,
                    _In_reads_(num_op_options) const char* const* ep_option_vals,
                    size_t num_ep_options);

// ??? Should these have 'Get' in the name in case we wanted to add public setters in the future?
// OrtHardwareDevice accessors.
ORT_API(OrtHardwareDeviceType, HardwareDevice_Type, _In_ const OrtHardwareDevice* device);
ORT_API(int32_t, HardwareDevice_VendorId, _In_ const OrtHardwareDevice* device);
ORT_API(const char*, HardwareDevice_Vendor, _In_ const OrtHardwareDevice* device);
ORT_API(int32_t, HardwareDevice_BusId, _In_ const OrtHardwareDevice* device);
ORT_API(const OrtKeyValuePairs*, HardwareDevice_Metadata, _In_ const OrtHardwareDevice* device);

// OrtEpDevice accessors
ORT_API(const char*, EpDevice_EpName, _In_ const OrtEpDevice* ep_device);
ORT_API(const char*, EpDevice_EpVendor, _In_ const OrtEpDevice* ep_device);
ORT_API(const OrtKeyValuePairs*, EpDevice_EpMetadata, _In_ const OrtEpDevice* ep_device);
ORT_API(const OrtKeyValuePairs*, EpDevice_EpOptions, _In_ const OrtEpDevice* ep_device);
ORT_API(const OrtHardwareDevice*, EpDevice_Device, _In_ const OrtEpDevice* ep_device);

// SessionOptions accessors
ORT_API_STATUS_IMPL(SessionOptions_GetConfigOptions,
                    _In_ const OrtSessionOptions* session_options, _Out_ OrtKeyValuePairs** options);
ORT_API(const char*, SessionOptions_GetConfigOption,
        _In_ const OrtSessionOptions* session_options, _In_ const char* key);
ORT_API(GraphOptimizationLevel, SessionOptions_GetOptimizationLevel, _In_ const OrtSessionOptions* session_options);

}  // namespace OrtExecutionProviderApi
