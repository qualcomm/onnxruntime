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
    Initialize = InitializeImpl;
    Shutdown = ShutdownImpl;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetExecutionDevices = GetExecutionDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static OrtStatus* InitializeImpl() noexcept {
    return nullptr;  // no initialization needed
  }

  static OrtStatus* ShutdownImpl() noexcept {
    return nullptr;  // no shutdown needed
  }

  static const char* GetNameImpl(OrtEpFactory* this_ptr) {
    DummyEpFactory* ep = static_cast<DummyEpFactory*>(this_ptr);
    return ep->ep_name_.c_str();
  }

  static const char* GetVendorImpl(OrtEpFactory* this_ptr) {
    DummyEpFactory* ep = static_cast<DummyEpFactory*>(this_ptr);
    return ep->vendor_.c_str();
  }

  static OrtStatus* GetExecutionDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice** devices,
                                            size_t num_devices,
                                            OrtExecutionDevice** execution_devices,
                                            size_t* num_execution_devices) {
    DummyEpFactory* ep = static_cast<DummyEpFactory*>(this_ptr);
    *execution_devices = nullptr;
    *num_execution_devices = 0;

    gsl::span<const OrtHardwareDevice*> devices_span{devices, num_devices};
    for (const auto* device : devices_span) {
      if (device == nullptr) {
        continue;  // should never happen
      }

      // FIXME: OrtHardwareDevice needs its own accessors
      if (ep->ep_api.HardwareDevice_Type(device) == OrtHardwareDeviceType::CPU) {
        OrtExecutionDevice* execution_device = nullptr;
        std::vector<const char*> keys;
        std::vector<const char*> values;
        // random example using made up options
        keys.push_back("big_cores");
        values.push_back("2");
        keys.push_back("little_cores");
        values.push_back("4");
        RETURN_IF_ERROR(
            ep->ep_api.CreateExecutionDevice(this_ptr, device, keys.data(), values.data(), keys.size(),
                                             &execution_device));

        *execution_devices = execution_device;  // ORT takes ownership
        *num_execution_devices = 1;

        // only one. implementation could potentially have multiple devices
        return nullptr;
      }
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* this_ptr, const OrtExecutionDevice* /*execution_device*/,
                                 const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp** ep) {
    // Create the execution provider
    DummyEpFactory* factory = static_cast<DummyEpFactory*>(this_ptr);
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Dummy EP", ORT_FILE, __LINE__, __FUNCTION__));

    OrtKeyValuePairs* options = nullptr;
    RETURN_IF_ERROR(factory->ep_api.SessionOptionsConfigOptions(session_options, &options));

    // look for any options using this prefix
    const std::string ep_options_prefix = "ep." + factory->ep_name_ + ".";

    // use properties from execution_device if needed

    auto dummy_ep = std::make_unique<DummyEp>(*factory, *options);
    factory->ort_api.ReleaseKeyValuePairs(options);

    *ep = dummy_ep.release();
    return nullptr;
  }

  static void ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) {
    DummyEpFactory* factory = static_cast<DummyEpFactory*>(this_ptr);

    // Release the execution provider
    for (auto& ep_instance : factory->eps_) {
      if (ep_instance.get() == ep) {
        ep_instance.reset();  // don't bother removing from vector as this point. probably doesn't matter.
        break;
      }
    }
  }

  const std::string ep_name_;                  // EP name library was registered with
  const std::string vendor_{"Contoso"};        // EP vendor name
  std::vector<std::unique_ptr<DummyEp>> eps_;  // EP instances created by this factory
};

//
// Public symbol
//
OrtEpFactory* GetEpFactory(const char* ep_name, const OrtApiBase* ort_api_base) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  static DummyEpFactory ep_plugin(ep_name, ApiPtrs{*ort_api, *ep_api});

  return &ep_plugin;
}
