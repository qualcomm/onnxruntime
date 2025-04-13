// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

OrtStatus* CreateEpFactory(const char* registration_name, const OrtApiBase* ort_api_base, OrtEpApi::OrtEpFactory** factory);
OrtStatus* ReleaseEpFactory(OrtEpApi::OrtEpFactory* factory);
