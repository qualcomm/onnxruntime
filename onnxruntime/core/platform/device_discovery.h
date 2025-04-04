// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "core/common/basic_types.h"
namespace onnxruntime {

class DeviceDiscovery {
 public:
  static std::vector<HardwareDevice>& GetDevices() {
    // assumption: devices don't change. we assume the machine must be shutdown to change cpu/gpu/npu devices.
    // technically someone could disable/enable a device in a running OS. we choose not to add complexity to support
    // that scenario.
    static std::vector<HardwareDevice> devices = DiscoverDevicesForPlatform();
    return devices;
  }

 private:
  DeviceDiscovery() = default;
  // platform specific code implements this method
  static std ::vector<HardwareDevice> DiscoverDevicesForPlatform();
};
}  // namespace onnxruntime
