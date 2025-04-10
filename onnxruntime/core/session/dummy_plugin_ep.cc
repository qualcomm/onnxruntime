#include "dummy_plugin_ep.h"
#include "onnxruntime_cxx_api.h"

#include <memory>
#include <string>
#include <vector>

#include <gsl/span>

using OrtEpFactory = OrtEpApi::OrtEpFactory;
using OrtEp = OrtEpApi::OrtEp;

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct DummyEp : OrtEp, ApiPtrs {
  DummyEp(ApiPtrs apis, OrtKeyValuePairs& /*config_options*/) : ApiPtrs(apis) {
    // Initialize the execution provider

    // TODO: Get EP specific settings out of config_options.
    // EP should copy any values it needs from options as factory will release the OrtKeyValuePairs instance which may
    // result in the const char* key/value instances becoming invalid.
  }

  ~DummyEp() {
    // Clean up the execution provider
  }

  std::string name_;
};

struct DummyEpFactory : OrtEpFactory, ApiPtrs {
  DummyEpFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetDeviceInfoIfSupported = GetDeviceInfoIfSupportedImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static const char* GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* ep = static_cast<const DummyEpFactory*>(this_ptr);
    return ep->ep_name_.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* ep = static_cast<const DummyEpFactory*>(this_ptr);
    return ep->vendor_.c_str();
  }

  static bool GetDeviceInfoIfSupportedImpl(const OrtEpFactory* this_ptr,
                                           const OrtHardwareDevice* device,
                                           _Out_opt_ OrtKeyValuePairs** ep_metadata,
                                           _Out_opt_ OrtKeyValuePairs** ep_options) {
    const auto* ep = static_cast<const DummyEpFactory*>(this_ptr);

    if (ep->ep_api.HardwareDevice_Type(device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // these can be returned as nullptr if you have nothing to add.
      ep->ort_api.CreateKeyValuePairs(ep_metadata);
      ep->ort_api.CreateKeyValuePairs(ep_options);

      // random example using made up values
      ep->ort_api.AddKeyValuePair(*ep_metadata, "version", "0.1");
      ep->ort_api.AddKeyValuePair(*ep_options, "run_really_fast", "true");

      return true;
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* this_ptr,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                 _In_ size_t num_devices,
                                 _In_ const OrtSessionOptions* session_options,
                                 _In_ const OrtLogger* logger,
                                 _Out_ OrtEp** ep) {
    auto* factory = static_cast<DummyEpFactory*>(this_ptr);

    if (num_devices != 1) {
      // we only registered for CPU and only expected to be selected for one CPU
      // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
      // the EP has been selected for.
      return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                           "Dummy EP only supports selection for one device.");
    }

    // Create the execution provider
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Dummy EP", ORT_FILE, __LINE__, __FUNCTION__));

    OrtKeyValuePairs* options = nullptr;
    RETURN_IF_ERROR(factory->ep_api.SessionOptionsConfigOptions(session_options, &options));

    // look for any config options in session_options using this prefix.
    // values returned from GetDeviceInfoIfSupportedImpl in ep_options have been added to session_options
    // along with any user provided values/overrides so that they are all in one place.
    const std::string ep_options_prefix = "ep." + factory->ep_name_ + ".";

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata_pairs[0];

    auto dummy_ep = std::make_unique<DummyEp>(*factory, *options);
    factory->ort_api.ReleaseKeyValuePairs(options);

    *ep = dummy_ep.release();
    return nullptr;
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    DummyEp* dummy_ep = static_cast<DummyEp*>(ep);
    delete dummy_ep;
  }

  const std::string ep_name_;            // EP name library was registered with
  const std::string vendor_{"Contoso"};  // EP vendor name
};

//
// Public symbols
//
OrtStatus* CreateEpFactory(const char* ep_name, const OrtApiBase* ort_api_base, OrtEpFactory** factory) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  // we just use a shared static instance.
  // implementation is also free to allocate on a per-call basis as ReleaseEpFactory will be called
  // when the EP is no longer needed.
  static DummyEpFactory ep_plugin(ep_name, ApiPtrs{*ort_api, *ep_api});
  *factory = &ep_plugin;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* /*factory*/) {
  // no-op for us
  return nullptr;
}
