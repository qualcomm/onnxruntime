// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace onnxruntime {

/** A computed hash value. */
using HashValue = uint64_t;

/** The type of an argument (input or output).*/
enum class ArgType : uint8_t {
  kInput,
  kOutput,
};

struct HardwareDevice {
  enum Type {
    CPU,
    GPU,
    NPU
  };

  // we always need to check vendor and type so make those mandatory properties
  std::string vendor;
  Type type;
  std::unordered_map<std::string, std::string> properties;
};

}  // namespace onnxruntime
