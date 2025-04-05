// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// Wrapper for OrtEpApi::OrtEp -> IExecutionProvider
class PluginEp : public IExecutionProvider {
 public:
  PluginEp(OrtEpApi::OrtEp& ep)
      : IExecutionProvider(ep.GetName(&ep)), ep_{ep} {
  }

 private:
  OrtEpApi::OrtEp& ep_;
};
}  // namespace onnxruntime
