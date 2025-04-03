#include "dummy_plugin_ep.h"

#include <memory>
#include <string>
#include <vector>

using OrtEpPlugin = OrtEpApi::OrtEpPlugin;
using OrtEpFactory = OrtEpApi::OrtEpFactory;
using OrtEp = OrtEpApi::OrtEp;

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct DummyEp : OrtEp, ApiPtrs {
  DummyEp(ApiPtrs apis, OrtKeyValuePairs& /*config_options*/) : ApiPtrs(apis) {
    // Initialize the execution provider
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
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

    return nullptr;
  }

  std::string name_;
  std::string vendor_;
};

struct DummyEpFactory : OrtEpFactory, ApiPtrs {
  DummyEpFactory(ApiPtrs apis) : ApiPtrs(apis) {
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl

    // Function ORT calls to release an EP instance.
    // void(ORT_API_CALL * ReleaseEP)(OrtEpFactory * this_ptr, OrtEp * ep);
  }

  ~DummyEpFactory() {
    // Clean up the factory
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* this_ptr, const OrtSessionOptions* session_options,
                                 const OrtLogger* logger, OrtEp** ep) {
    // Create the execution provider
    DummyEpFactory* factory = static_cast<DummyEpFactory*>(this_ptr);

    OrtKeyValuePairs* options = nullptr;
    auto status = factory->ep_api.SessionOptionsConfigOptions(session_options, /*filter*/ nullptr, &options);

    if (status != nullptr) {
      return status;
    }

    auto ep = std::make_unique<DummyEp>(factory->ort_api, *options);
    factory->ort_api.ReleaseKeyValuePairs(options);

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
    data = nullptr;  // state if needed. as we have a this pointer that seems unnecessary
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

OrtStatus* CreateEpPlugins(const OrtApiBase* ort_api_base, /*out*/ OrtEpApi::OrtEpPlugin** out, size_t num_eps) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();

  // for each EP this library implements register a factory function
  ep_api->RegisterEpPlugin(*ort_api, *ep_api);
}

OrtStatus* ReleaseEpPlugin(OrtEpApi::OrtEpPlugin* ep_plugin) {
}
