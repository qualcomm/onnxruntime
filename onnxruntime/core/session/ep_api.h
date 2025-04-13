// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <mutex>

#include "core/common/common.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"

namespace OrtExecutionProviderApi {

// implementation that returns the API struct
ORT_API(const OrtEpApi*, GetEpApi);

ORT_API_STATUS_IMPL(RegisterExecutionProviderLibrary, _In_ OrtEnv* env, const char* ep_name, const ORTCHAR_T* path);
ORT_API_STATUS_IMPL(UnregisterExecutionProviderLibrary, _In_ OrtEnv* env, _In_ const char* ep_name);

// OrtHardwareDevice and OrtExecutionDevice accessors
ORT_API(OrtHardwareDeviceType, HardwareDevice_Type, _In_ const OrtHardwareDevice* device);
ORT_API(int32_t, HardwareDevice_VendorId, _In_ const OrtHardwareDevice* device);
ORT_API(const char*, HardwareDevice_Vendor, _In_ const OrtHardwareDevice* device);
ORT_API(int32_t, HardwareDevice_BusId, _In_ const OrtHardwareDevice* device);
ORT_API(const OrtKeyValuePairs*, HardwareDevice_Properties, _In_ const OrtHardwareDevice* device);

// ORT_API(const char*, ExecutionDevice_EpName, _In_ const OrtExecutionDevice* device);
// ORT_API(const char*, ExecutionDevice_EpVendor, _In_ const OrtExecutionDevice* device);
// ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpMetadata, _In_ const OrtExecutionDevice* device);
// ORT_API(const OrtKeyValuePairs*, ExecutionDevice_EpOptions, _In_ const OrtExecutionDevice* device);
// ORT_API(const OrtHardwareDevice*, ExecutionDevice_Device, _In_ const OrtExecutionDevice* device);

// user must call ReleaseKeyValuePairs when done.
ORT_API_STATUS_IMPL(SessionOptionsConfigOptions, _In_ const OrtSessionOptions* session_options,
                    _Out_ OrtKeyValuePairs** options);

// Get ConfigOptions by key. Returns null in value if key not found (vs pointer to empty string if found).
ORT_API(const char*, SessionOptionsConfigOption, _In_ const OrtSessionOptions* session_options, _In_ const char* key);

}  // namespace OrtExecutionProviderApi
