// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/utils.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

using namespace onnxruntime;

common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size) {
  const size_t str_len = str.size();
  const size_t req_size = str_len + 1;

  if (out == nullptr) {  // User is querying the total output buffer size
    *size = req_size;
    return onnxruntime::common::Status::OK();
  }

  if (*size >= req_size) {  // User provided a buffer of sufficient size
    std::memcpy(out, str.data(), str_len);
    out[str_len] = '\0';
    *size = req_size;
    return onnxruntime::common::Status::OK();
  }

  // User has provided a buffer that is not large enough
  *size = req_size;
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, err_msg);
}

// provider either model_path, or modal_data + model_data_length.
OrtStatus* CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                     _In_ const OrtEnv* env,
                                     _In_opt_z_ const ORTCHAR_T* model_path,
                                     _In_opt_ const void* model_data,
                                     size_t model_data_length,
                                     std::unique_ptr<onnxruntime::InferenceSession>& sess) {
  // quick check here to decide load path. InferenceSession will provide error message for invalid values.
  // TODO: Could move to a helper
  const Env& os_env = Env::Default();  // OS environment (!= ORT environment)
  bool load_config_from_model =
      os_env.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar) == "1";

  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    if (model_path != nullptr) {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_path);
    } else {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_data, static_cast<int>(model_data_length));
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Loading config from ONNX models is not supported in this build.");
#endif
  } else {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment());
  }

  // create EPs now that we have the session and session logger
  // TODO: we need a mutable Env but adding a whole new set of things at the API level is a lot.
  //       figure out the best way to support this. can we split out session creation from EP init and model load?
  const_cast<OrtEnv*>(env)->GetEnvironment().CreateExecutionProviders(*options, *sess);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Add custom domains
  if (options && !options->custom_op_domains_.empty()) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddCustomOpDomains(options->custom_op_domains_));
  }
#endif

  // Finish load
  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load());
#endif
  } else {
    if (model_path != nullptr) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_path));
    } else {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_data, static_cast<int>(model_data_length)));
    }
  }

  return nullptr;
}

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ onnxruntime::InferenceSession& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container) {
  // TODO: If the session is using autoep selection, use the environment to do selection
  // The OrtExecutionDevice entries are in ep_libraries_;

  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      provider_list.push_back(std::move(provider));
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess.RegisterExecutionProvider(std::move(provider)));
    }
  }

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess.AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess.Initialize());

  return nullptr;
}

namespace onnxruntime {
// Select execution providers based on the device policy and available devices and add to session
// TODO: Should this be in session or lower like framework?
Status SelectEPs(const Environment& env, OrtExecutionProviderDevicePolicy device_policy,
                 InferenceSession& sess, const OrtLogger& logger) {
  struct SelectionInfo {
    OrtEpApi::OrtEpFactory* ep_factory;
    std::vector<const OrtHardwareDevice*> devices;
    std::vector<const OrtKeyValuePairs*> ep_metadata;
  };

  std::unordered_map<std::string, SelectionInfo> eps_selected;
  const auto add_selection = [&eps_selected, &sess](const OrtExecutionDevice& ed) -> Status {
    auto iter = eps_selected.find(ed.ep_name);
    if (iter == eps_selected.end()) {
      eps_selected[ed.ep_name] = SelectionInfo{ed.ep_factory, {ed.device}, {&ed.ep_metadata}};
    } else {
      ORT_ENFORCE(iter->second.ep_factory == ed.ep_factory, "Inconsistent factory pointers. EP:", ed.ep_name);
      iter->second.devices.push_back(ed.device);
      iter->second.ep_metadata.push_back(&ed.ep_metadata);
    }

    auto& config_options = sess.GetMutableSessionOptions().config_options;
    for (const auto& entry : ed.ep_options.entries) {
      // preserve user-provided options as they override any defaults the EP factory specified earlier
      if (config_options.configurations.find(entry.first) == config_options.configurations.end()) {
        // use AddConfigEntry for the error checking it does
        ORT_RETURN_IF_ERROR(config_options.AddConfigEntry(entry.first.c_str(), entry.second.c_str()));
      }
    }

    return Status::OK();
  };

  const auto& execution_devices = env.GetExecutionDevices();

  if (device_policy == OrtExecutionProviderDevicePolicy_PREFER_CPU) {
    // pick first CPU option for now
    for (const OrtExecutionDevice* ed : execution_devices) {
      if (ed->device->type == OrtHardwareDeviceType_CPU) {
        ORT_RETURN_IF_ERROR(add_selection(*ed));
        break;
      }
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Device policy not implemented yet: ", static_cast<int>(device_policy));
  }

  if (eps_selected.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "No execution providers selected. Please check the device policy and available devices.");
  }

  // create OrtSessionOptions as that's what the EPs need to use
  OrtSessionOptions ort_so;
  // this copy isn't ideal but we need the user provided options which were in a const OrtSesionOptions in the
  // call to CreateSessionAndLoadModel. I don't think we can make those non-const as existing users may re-use
  // SessionOptions to create multiple sessions current and doing so would break that.
  // Consider updating the OrtSessionOptions implementation to support `value` and an optional
  // const SessionOptions* that can be preferred for reading values if it exists.
  ort_so.value = sess.GetSessionOptions();

  for (const auto& entry : eps_selected) {
    OrtEpApi::OrtEp* ep = nullptr;
    // add the ep_options to session options but leave any existing entries (user provided overrides) untouched.
    const SelectionInfo& info = entry.second;
    auto status = info.ep_factory->CreateEp(info.ep_factory, info.devices.data(), info.ep_metadata.data(),
                                            info.devices.size(), &ort_so, &logger, &ep);
    if (status != nullptr) {
      return ToStatus(status);
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
