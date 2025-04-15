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

// SessionOptions accessors
ORT_API_STATUS_IMPL(SessionOptions_GetConfigOptions,
                    _In_ const OrtSessionOptions* session_options, _Out_ OrtKeyValuePairs** options);
ORT_API(const char*, SessionOptions_GetConfigOption,
        _In_ const OrtSessionOptions* session_options, _In_ const char* key);
ORT_API(GraphOptimizationLevel, SessionOptions_GetOptimizationLevel, _In_ const OrtSessionOptions* session_options);

}  // namespace OrtExecutionProviderApi
