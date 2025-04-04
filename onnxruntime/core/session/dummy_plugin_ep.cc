#include "dummy_plugin_ep.h"
#include "onnxruntime_cxx_api.h"

#include <memory>
#include <string>
#include <vector>

#include <gsl/span>

using OrtEpPlugin = OrtEpApi::OrtEpPlugin;
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
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;

    // TODO: Get EP specific settings out of config_options.
    // EP should copy any values it needs from options as factory will release the OrtKeyValuePairs instance which may
    // result in the const char* key/value instances becoming invalid.
  }

  ~DummyEp() {
    // Clean up the execution provider
  }

  static const char* GetNameImpl(OrtEp* this_ptr) {
    DummyEp* ep = static_cast<DummyEp*>(this_ptr);
    return ep->name_.c_str();
  }

  static const char* GetVendorImpl(OrtEp* this_ptr) {
    DummyEp* ep = static_cast<DummyEp*>(this_ptr);
    return ep->vendor_.c_str();
  }

  static OrtStatus* GetExecutionDevicesImpl(OrtEp* this_ptr,
                                            const OrtHardwareDevice** devices,
                                            size_t num_devices,
                                            OrtExecutionDevice** execution_devices,
                                            size_t* num_execution_devices) {
    DummyEp* ep = static_cast<DummyEp*>(this_ptr);
    *execution_devices = nullptr;
    *num_execution_devices = 0;

    gsl::span<const OrtHardwareDevice*> devices_span{devices, num_devices};
    for (const auto* device : devices_span) {
      if (device == nullptr) {
        continue;  // should never happen
      }

      if (device->type == OrtHardwareDeviceType::CPU) {
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

        *execution_devices = execution_device;
        *num_execution_devices = 1;

        // only one. implementation could potentially have multiple devices
        return nullptr;
      }
    }

    return nullptr;
  }

  std::string name_;
  std::string vendor_;
};

struct DummyEpFactory : OrtEpFactory, ApiPtrs {
  DummyEpFactory(ApiPtrs apis) : ApiPtrs(apis) {
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* this_ptr, const OrtSessionOptions* session_options,
                                 const OrtLogger* logger, OrtEp** ep) {
    // Create the execution provider
    DummyEpFactory* factory = static_cast<DummyEpFactory*>(this_ptr);
    factory->ort_api.Logger_LogMessage(logger,
                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                       "Creating Dummy EP", ORT_FILE, __LINE__, __FUNCTION__);

    OrtKeyValuePairs* options = nullptr;
    RETURN_IF_ERROR(factory->ep_api.SessionOptionsConfigOptions(session_options, &options));
    auto dummy_ep = std::make_unique<DummyEp>(*factory, *options);
    factory->ort_api.ReleaseKeyValuePairs(options);

    *ep = dummy_ep.release();
    nullptr;
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

  OrtStatus* GetExecutionProviderName(const char** name) {
    *name = "DummyPluginEP";
    return nullptr;
  }

  std::vector<std::unique_ptr<DummyEp>> eps_;
};

struct DummyPluginEp : OrtEpApi::OrtEpPlugin, ApiPtrs {
  DummyPluginEp(ApiPtrs apis) : ApiPtrs{apis} {
    // Initialize the plugin
    CreateEpFactory = CreateEpFactoryImpl;
    ReleaseEpFactory = ReleaseEpFactoryImpl;
  }

  ~DummyPluginEp() {
    // Clean up the plugin
  }

  static OrtStatus* CreateEpFactoryImpl(OrtEpPlugin* this_ptr, OrtEpFactory** ep_factory) {
    DummyPluginEp* plugin = static_cast<DummyPluginEp*>(this_ptr);

    auto factory = std::make_unique<DummyEpFactory>(*plugin);
    *ep_factory = factory.get();
    plugin->factories_.push_back(std::move(factory));

    return nullptr;
  }

  static void ReleaseEpFactoryImpl(OrtEpPlugin* this_ptr, OrtEpFactory* ep_factory) {
    // Release the execution provider
    DummyPluginEp* plugin = static_cast<DummyPluginEp*>(this_ptr);

    for (auto& factory : plugin->factories_) {
      if (factory.get() == ep_factory) {
        factory.reset();  // don't bother removing from vector as this point. probably doesn't matter.
        break;
      }
    }
  }

  std::vector<std::unique_ptr<DummyEpFactory>> factories_;
};

//
// Public symbols
//

OrtStatus* CreateEpPlugins(const OrtApiBase* ort_api_base) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();

  // for each EP this library implements register a factory function
  auto plugin = std::make_unique<DummyPluginEp>(ApiPtrs{*ort_api, *ep_api});
  ep_api->RegisterEpPlugin(plugin.release());

  return nullptr;
}

OrtStatus* ReleaseEpPlugin(OrtEpApi::OrtEpPlugin* ep_plugin) {
  DummyPluginEp* plugin = static_cast<DummyPluginEp*>(ep_plugin);
  delete plugin;

  return nullptr;
}
