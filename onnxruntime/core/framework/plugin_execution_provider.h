// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// Wrapper for OrtEpApi::OrtEp -> IExecutionProvider
class PluginEp : public IExecutionProvider {
 public:
  PluginEp(OrtEpApi::OrtEpFactory& factory, OrtEpApi::OrtEp& ep)
      : IExecutionProvider(ep.GetName(&ep)), ep_factory_{factory}, ep_{ep} {
  }

  ~PluginEp() {
    ep_factory_.ReleaseEp(&ep_factory_, &ep_);
  }

 private:
  OrtEpApi::OrtEpFactory& ep_factory_;
  OrtEpApi::OrtEp& ep_;
};
}  // namespace onnxruntime
