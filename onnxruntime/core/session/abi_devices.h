// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/session/abi_key_value_pairs.h"
#include "core/session/onnxruntime_c_api.h"  // FIXME: for OrtEpApi::OrtEpFactory but we need a better way
// struct OrtEpApi;
// struct OrtEpApi::OrtEpFactory;

struct OrtHardwareDevice {
  enum Type {
    CPU,
    GPU,
    NPU
  };

  // we always need to check vendor and type so make those mandatory properties
  std::string vendor;
  Type type;
  OrtKeyValuePairs properties;
};

// EP + HardwareDevice.
struct OrtExecutionDevice {
  std::string ep_name;
  std::string ep_vendor;
  const OrtHardwareDevice* device;  // TODO: copy or reference. start with assumption device list is static so reference

  // metadata from EP
  OrtKeyValuePairs properties;
  OrtEpApi::OrtEpFactory* ep_factory;  // FIXME: figure out how to have this type here
};
