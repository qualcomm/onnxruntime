// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

//OrtStatus* CreateEpPlugins(const OrtApiBase* ort_api_base);
//OrtStatus* ReleaseEpPlugin(OrtEpApi::OrtEpPlugin* ep_plugin);
OrtStatus* CreateEpFactories(const OrtApiBase* ort_api_base);
OrtStatus* ReleaseEpFactory(OrtEpApi::OrtEpFactory* ep_factory);
