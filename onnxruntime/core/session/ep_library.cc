// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library.h"

#include "core/framework/provider_options.h"
#include "core/framework/session_options.h"

namespace onnxruntime {
// many EPs parse the options from prior to them being added to session options.
// to support that we need to extract the EP specific options from session_options and remove the prefix.
ProviderOptions EpLibrary::GetOptionsFromSessionOptions(const std::string& ep_name,
                                                        const SessionOptions& session_options) {
  const std::string option_prefix = ProviderOptionsUtils::GetProviderOptionPrefix(ep_name);
  ProviderOptions ep_options;

  for (const auto& [key, value] : session_options.config_options.configurations) {
    if (key.find(option_prefix) == 0) {
      // remove the prefix and add
      ep_options[key.substr(option_prefix.length())] = value;
    }
  }

  return ep_options;
}
}  // namespace onnxruntime
