// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtExecutionProviderApi {

// implementation that returns the API struct
ORT_API(const OrtEpApi*, GetEpApi);

ORT_API_STATUS_IMPL(RegisterEpFactory, _In_ OrtEnv* env, OrtEpApi::OrtEpFactory* factory);
ORT_API_STATUS_IMPL(RegisterLegacyEpFactory, _In_ OrtEnv* env, const ORTCHAR_T* library_path);

ORT_API_STATUS_IMPL(CreateExecutionDevice, _In_ /*const*/ OrtEpApi::OrtEp* ep,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_reads_(num_ep_device_properties) const char** ep_device_properties_keys,
                    _In_reads_(num_ep_device_properties) const char** ep_device_properties_values,
                    _In_ size_t num_ep_device_properties,
                    _Out_ OrtExecutionDevice** ort_execution_device);

ORT_API_STATUS_IMPL(SessionOptionsConfigOptions, _In_ const OrtSessionOptions* session_options,
                    _Out_ OrtKeyValuePairs** options);

// Get ConfigOptions by key. Returns null in value if key not found (vs pointer to empty string if found).
ORT_API(const char*, SessionOptionsConfigOption, _In_ const OrtSessionOptions* session_options, _In_ const char* key);

}  // namespace OrtExecutionProviderApi
