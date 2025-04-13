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
Status AutoSelectEPs(const Environment& env, InferenceSession& sess, const std::string& ep_to_select) {
  const auto& execution_devices = env.GetExecutionDevices();

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
  const OrtLogger* api_session_logger = session_logger->ToExternal();

  // CPU
  for (const auto* device : execution_devices) {
    if (device->ep_name != ep_to_select) {
      continue;
    }

    InternalEpFactory* internal_factory = nullptr;
    if (device->ep_name == kCpuExecutionProvider) {
      if (ep_to_select == kCpuExecutionProvider) {
        internal_factory = env.GetInternalEpFactory(device->ep_factory);
      }
    } else if (device->ep_name == kDmlExecutionProvider) {
    }

    // in the real implementation multiple devices can be assigned to an EP
    std::vector<const OrtHardwareDevice*> devices{device->device};
    std::vector<const OrtKeyValuePairs*> ep_metadata{&device->ep_metadata};

    std::shared_ptr<IExecutionProvider> ep;
    if (internal_factory) {
      // this is a factory we created and registered
      OrtStatus* status = internal_factory->CreateIExecutionProvider(devices.data(), ep_metadata.data(),
                                                                     devices.size(), &ort_so, api_session_logger, ep);
      if (status != nullptr) {
        return ToStatus(status);
      }
    } else {
      OrtEpApi::OrtEp* api_ep = nullptr;
      // add the ep_options to session options but leave any existing entries (user provided overrides) untouched.
      auto status = device->ep_factory->CreateEp(device->ep_factory, devices.data(), ep_metadata.data(),
                                                 devices.size(), &ort_so, api_session_logger, &api_ep);
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
  // TEMPORARY for testing. No providers registered equates to defaulting to ORT CPU EP currently so we need to honour
  // that in the real world.
  auto auto_select_ep_name = sess->GetSessionOptions().config_options.GetConfigEntry("test.ep_to_select");
  if (auto_select_ep_name) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(AutoSelectEPs(env->GetEnvironment(), *sess, *auto_select_ep_name));
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
  const logging::Logger* session_logger = sess.GetLogger();
  ORT_ENFORCE(session_logger != nullptr,
              "Session logger is invalid, but should have been initialized during session construction.");

  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider(*options, *session_logger->ToExternal());
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
