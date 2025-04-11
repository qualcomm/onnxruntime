// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/utils.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/inference_session_utils.h"
#include "core/session/internal_ep_factory.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"

using namespace onnxruntime;
namespace {
// Select execution providers based on the device policy and available devices and add to session
// TODO: Should this be in session or lower like framework?
Status AutoSelectEPs(const Environment& env, const OrtSessionOptions* options, InferenceSession& sess) {
  struct SelectionInfo {
    OrtEpApi::OrtEpFactory* ep_factory;
    std::vector<const OrtHardwareDevice*> devices;
    std::vector<const OrtKeyValuePairs*> ep_metadata;
  };

  // Initial idea is that we have at most 1 NPU, 1 GPU, _maybe_ 2 CPU if ORT is a fallback to an IHV CPU EP to ensure
  // we cover opsets and operators that the IHV CPU EP does not support. TBD if that is required, but for now:
  // Indexes are as follows.
  // 0: NPU EP name
  // 1: GPU EP name
  // 2: CPU EP (excluding ORT)
  // 3: ORT CPU EP
  std::array<std::string, 4> ep_priority_order;

  std::unordered_map<std::string, SelectionInfo> eps_selected;
  const auto add_selection = [&eps_selected, &sess, &ep_priority_order](const OrtExecutionDevice& ed) -> Status {
    switch (ed.device->type) {
      case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
        assert(ep_priority_order[0].empty());
        ep_priority_order[0] = ed.ep_name;
        break;
      case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
        assert(ep_priority_order[1].empty());
        ep_priority_order[1] = ed.ep_name;
        break;
      case OrtHardwareDeviceType::OrtHardwareDeviceType_CPU:
        if (ed.ep_name != kCpuExecutionProvider) {
          assert(ep_priority_order[2].empty());
          ep_priority_order[2] = ed.ep_name;
        } else {
          ep_priority_order[3] = ed.ep_name;
        }
        break;
    }

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

  ORT_ENFORCE(options->ep_selection_policy.delegate == nullptr,
              "EP selection delegate support is not implemented yet.");
  ORT_ENFORCE(options->ep_selection_policy.policy == OrtExecutionProviderDevicePolicy_PREFER_CPU,
              "Only OrtExecutionProviderDevicePolicy_DEFAULT policy is currently implemented.");
  if (options->ep_selection_policy.policy == OrtExecutionProviderDevicePolicy_PREFER_CPU) {
    // pick first CPU option for now
    for (const OrtExecutionDevice* ed : execution_devices) {
      if (ed->device->type == OrtHardwareDeviceType_CPU) {
        ORT_RETURN_IF_ERROR(add_selection(*ed));
        break;
      }
    }
  }

  if (eps_selected.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "No execution providers selected. Please check the device policy and available devices.");
  }

  // FIXME: Is this too fragile? The EP can't keep a pointer to or read from the session options the CreateEp
  // gets once that call returns as it's pointing to this local variable.

  // create OrtSessionOptions for the CreateEp call.
  // Once the InferenceSession is created, its SessionOptions is the source of truth.
  // We added the ep_options there as well, and need to pass those values through.
  OrtSessionOptions ort_so;
  // this copy isn't ideal, but we need the user provided options which were in a const OrtSesionOptions in the
  // call to CreateSessionAndLoadModel, and are now in the InferenceSession. it shouldn't be too heavyweight.
  // if we need to optimize we could add an `onnxruntime::SessionOptions* existing` member to OrtSessionOptions,
  // set it to the InferenceSession's SessionOptions, and prefer that over OrtSessionOptions.value in accessors
  // like OrtEpApi::SessionOptionsConfigOptions.
  // the CreateEp call would be the only user and as it gets a const OrtSessionOptions* it can only use accessors.
  ort_so.value = sess.GetSessionOptions();
  const auto& session_logger = sess.GetLogger();
  // OrtLogger is a cast from logging::Logger
  const OrtLogger* api_session_logger = reinterpret_cast<const OrtLogger*>(&session_logger);

  bool disable_ort_cpu_ep = ort_so.value.config_options.GetConfigEntry(kOrtSessionOptionsDisableCPUEPFallback) == "1";
  size_t end = ep_priority_order.size();
  if (disable_ort_cpu_ep) {
    // skip the ORT CPU EP if the user has disabled it. that's always in the last slot.
    --end;
  }

  for (size_t idx = 0; idx < end; ++idx) {
    const auto& ep_name = ep_priority_order[idx];
    if (ep_name.empty()) {
      continue;
    }

    const auto& info = eps_selected[ep_name];
    InternalEpFactory* internal_factory = env.GetInternalEpFactory(info.ep_factory);
    std::shared_ptr<IExecutionProvider> ep;
    if (internal_factory) {
      // this is a factory we created and registered
      OrtStatus* status = internal_factory->CreateIExecutionProvider(info.devices.data(), info.ep_metadata.data(),
                                                                     info.devices.size(), &ort_so, api_session_logger,
                                                                     ep);
      if (status != nullptr) {
        return ToStatus(status);
      }
    } else {
      OrtEpApi::OrtEp* api_ep = nullptr;
      // add the ep_options to session options but leave any existing entries (user provided overrides) untouched.
      auto status = info.ep_factory->CreateEp(info.ep_factory, info.devices.data(), info.ep_metadata.data(),
                                              info.devices.size(), &ort_so, api_session_logger,
                                              &api_ep);
      if (status != nullptr) {
        return ToStatus(status);
      }
      // in the real setup we need an IExecutionProvider wrapper implementation that uses the OrtEp internally,
      // and we would add that IExecutionProvider to the InferenceSession.
      // that wrapper would also be responsible for the ReleaseEp call so needs a factory pointer as well.
      ORT_NOT_IMPLEMENTED("IExecutionProvider that wraps OrtEp has not been implemented.");
    }

    ORT_RETURN_IF_ERROR(sess.RegisterExecutionProvider(std::move(ep)));
  }

  return Status::OK();
}
}  // namespace

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

  // if there are no providers registered, and there's an ep selection policy set, do auto ep selection
  if (options->provider_factories.empty() &&
      options->ep_selection_policy.enable) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(AutoSelectEPs(env->GetEnvironment(), options, *sess));
  }

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
