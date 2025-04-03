// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

OrtStatus* CreateEpPlugins(const OrtApiBase* ort_api_base, /*out*/OrtEpApi::OrtEpPlugin** out, size_t num_eps);
OrtStatus* ReleaseEpPlugin(OrtEpApi::OrtEpPlugin* ep_plugin);
