// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/common/hash_combine.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/onnxruntime_c_api.h"  // OrtHardwareDeviceType and OrtEpApi::OrtEpFactory

struct OrtHardwareDevice {
  // we always need to check vendor and type so make those mandatory properties
  OrtHardwareDeviceType type;
  int32_t vendor_id;   // GPU has id
  std::string vendor;  // CPU uses string
  int32_t bus_id;
  OrtKeyValuePairs properties;

  static size_t Hash(const OrtHardwareDevice& hd) {
    auto h = std::hash<int>()(hd.vendor_id);  // start with something less trivial than the type
    onnxruntime::HashCombine(hd.vendor, h);
    onnxruntime::HashCombine(hd.bus_id, h);
    onnxruntime::HashCombine(hd.type, h);
    // skip the properties for now

    return h;
  }
};

// This is to make OrtDevice a valid key in hash tables
namespace std {
template <>
struct hash<OrtHardwareDevice> {
  size_t operator()(const OrtHardwareDevice& hd) const {
    return OrtHardwareDevice::Hash(hd);
  }
};

template <>
struct equal_to<OrtHardwareDevice> {
  bool operator()(const OrtHardwareDevice& lhs, const OrtHardwareDevice& rhs) const noexcept {
    return lhs.type == rhs.type &&
           lhs.vendor_id == rhs.vendor_id &&
           lhs.vendor == rhs.vendor &&
           lhs.bus_id == rhs.bus_id &&
           lhs.properties.keys == rhs.properties.keys &&
           lhs.properties.values == rhs.properties.values;
  }
};
}  // namespace std

struct OrtEpDevice {
  std::string ep_name;
  std::string ep_vendor;
  const OrtHardwareDevice* device;

  OrtKeyValuePairs ep_metadata;
  OrtKeyValuePairs ep_options;

  OrtEpApi::OrtEpFactory* ep_factory;
};
