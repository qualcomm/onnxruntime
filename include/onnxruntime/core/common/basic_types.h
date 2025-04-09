// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include "core/session/abi_key_value_pairs.h"

namespace onnxruntime {

/** A computed hash value. */
using HashValue = uint64_t;

/** The type of an argument (input or output).*/
enum class ArgType : uint8_t {
  kInput,
  kOutput,
};

}  // namespace onnxruntime

