// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/common/basic_types.h"

namespace onnxruntime {
// EP + HardwareDevice.
struct ExecutionDevice {
  std::string name;              // EP name
  std::string execution_vendor;  // EP vendor
  const HardwareDevice& device;  // TODO: copy or reference. start with assumption device list is static so reference

  // metadata from EP
  std::unordered_map<std::string, std::string> properties;
};
}  // namespace onnxruntime
