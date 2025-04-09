// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

OrtEpApi::OrtEpFactory* GetEpFactory(const char* ep_name, const OrtApiBase* ort_api_base);
