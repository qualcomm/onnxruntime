// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <array>
#include <cassert>
#include <codecvt>
#include <locale>
#include <string>
#include <unordered_set>

// GetDxgiInfo
// #include <dxgi.h>
// #pragma comment(lib, "dxgi.lib")
// #include <iostream>

#include "core/common/cpuid_info.h"
#include "core/session/abi_devices.h"

//// UsingSetupApi
#include <Windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cfgmgr32.h>
#pragma comment(lib, "setupapi.lib")
////

//// Using D3D12
// #include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
////

//// Using DXCore
// #define DXCORE_AVAILABLE (NTDDI_VERSION < NTDDI_WIN10_RS5)
#define DXCORE_AVAILABLE 1
#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wil/com.h>
////

namespace onnxruntime {
namespace {

// device info we accumulate from various sources
struct DeviceInfo {
  OrtHardwareDeviceType type;
  uint32_t vendor_id;
  uint32_t device_id;
  std::wstring vendor;
  std::wstring description;
  std::vector<DWORD> bus_ids;  // assuming could have multiple GPUs that are the same model
  std::unordered_map<std::wstring, std::wstring> metadata;
};

uint64_t GetDeviceKey(uint32_t vendor_id, uint32_t device_id) {
  return (uint64_t(vendor_id) << 32) | device_id;
}

uint64_t GetDeviceKey(const DeviceInfo& device_info) {
  return GetDeviceKey(device_info.vendor_id, device_info.device_id);
}

uint64_t GetLuidKey(LUID luid) {
  return (uint64_t(luid.HighPart) << 32) | luid.LowPart;
}

// key: hardware id with vendor and device id in it
// returns info for display and processor entries. key is (vendor_id << 32 | device_id)
// npus: (vendor_id << 32 | device_id) for devices we think are NPUs from DXCORE
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoSetupApi(const std::unordered_set<uint64_t>& npus) {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  std::array<GUID, 3> guids = {
      GUID_DEVCLASS_DISPLAY,
      GUID_DEVCLASS_PROCESSOR,
      GUID_DEVCLASS_SYSTEM,
  };

  for (auto guid : guids) {
    HDEVINFO devInfo = SetupDiGetClassDevs(&guid, nullptr, nullptr, DIGCF_PRESENT);
    if (devInfo == INVALID_HANDLE_VALUE) {
      return device_info;
    }

    SP_DEVINFO_DATA devData = {};
    devData.cbSize = sizeof(SP_DEVINFO_DATA);

    std::wstring buffer;
    buffer.resize(1024);

    for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &devData); ++i) {
      DWORD size = 0;
      DWORD regDataType = 0;

      uint64_t key;
      DeviceInfo* entry = nullptr;

      //// Get hardware ID (contains VEN_xxxx&DEV_xxxx)
      if (SetupDiGetDeviceRegistryPropertyW(devInfo,
                                            &devData,
                                            SPDRP_HARDWAREID,
                                            &regDataType,
                                            (PBYTE)buffer.data(),
                                            (DWORD)buffer.size(),
                                            &size)) {
        // std::wcout << L"Device Instance: " << buffer << std::endl;
        // PCI\VEN_xxxx&DEV_yyyy&...
        // ACPI\VEN_xxxx&DEV_yyyy&...
        // ACPI\VEN_
        // Include the root, \, and VEN_xxxx&DEV_yyyy
        // ??? can ACPI values be longer
        const auto get_id = [](const std::wstring& hardware_id, const std::wstring& prefix) -> uint32_t {
          if (auto idx = hardware_id.find(prefix); idx != std::wstring::npos) {
            auto id = hardware_id.substr(idx + prefix.size(), 4);
            if (std::all_of(id.begin(), id.end(), iswxdigit)) {
              return std::stoul(id, nullptr, 16);
            }
          }

          return 0;
        };

        uint32_t vendor_id = get_id(buffer, L"VEN_");
        uint32_t device_id = get_id(buffer, L"DEV_");
        // won't always have a vendor id from an ACPI entry. need at least a device id to identify the hardware
        if (vendor_id == 0 && device_id == 0) {
          continue;
        }

        key = GetDeviceKey(vendor_id, device_id);

        if (device_info.find(key) == device_info.end()) {
          device_info[key] = {};
        } else {
          if (guid == GUID_DEVCLASS_PROCESSOR) {
            // skip duplicate processor entries as we don't need to accumulate bus numbers for them
            continue;
          }
        }

        entry = &device_info[key];
        entry->vendor_id = vendor_id;
        entry->device_id = device_id;
      } else {
        continue;
      }

      // Get device description.
      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_DEVICEDESC, nullptr,
                                            (PBYTE)buffer.data(), (DWORD)buffer.size(), &size)) {
        entry->description = buffer;

        // Should we require the NPU to be found by DXCORE or do we want to allow this vague matching?
        const auto possible_npu = [](const std::wstring& desc) {
          return (desc.find(L"NPU") != std::wstring::npos ||
                  desc.find(L"Neural") != std::wstring::npos ||
                  desc.find(L"AI Engine") != std::wstring::npos ||
                  desc.find(L"VPU") != std::wstring::npos);
        };

        // not 100% accurate. is there a better way?
        uint64_t npu_key = GetDeviceKey(*entry);
        bool is_npu = npus.count(npu_key) > 0 || possible_npu(entry->description);

        if (guid == GUID_DEVCLASS_DISPLAY) {
          entry->type = OrtHardwareDeviceType_GPU;
        } else if (guid == GUID_DEVCLASS_PROCESSOR) {
          entry->type = is_npu ? OrtHardwareDeviceType_NPU : OrtHardwareDeviceType_CPU;
        } else if (guid == GUID_DEVCLASS_SYSTEM) {
          if (!is_npu) {
            // we're only iterating system devices to look for NPUs so drop anything else
            device_info.erase(key);
            continue;
          }

          entry->type = OrtHardwareDeviceType_NPU;
        } else {
          // unknown device type
          continue;
        }
      }

      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_MFG, nullptr,
                                            (PBYTE)buffer.data(), (DWORD)buffer.size(), &size)) {
        entry->vendor = buffer;
      }

      if (guid != GUID_DEVCLASS_PROCESSOR) {
        DWORD busNumber = 0;
        size = 0;
        if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_BUSNUMBER, nullptr,
                                              reinterpret_cast<PBYTE>(&busNumber), sizeof(busNumber), &size)) {
          // push_back in case there are two identical devices. not sure how else to tell them apart
          entry->bus_ids.push_back(busNumber);
        }
      }
    }

    SetupDiDestroyDeviceInfoList(devInfo);
  }

  return device_info;
}

// returns LUID to DeviceInfo
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoD3D12() {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  IDXGIFactory6* factory = nullptr;
  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
  if (FAILED(hr)) {
    std::cerr << "Failed to create DXGI factory.\n";
    return device_info;
  }

  IDXGIAdapter1* adapter = nullptr;

  // iterate by high-performance GPU preference first
  for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                       IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    do {
      if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
          (desc.Flags & DXGI_ADAPTER_FLAG_REMOTE) != 0) {
        // software or remote. skip
        break;
      }

      static_assert(sizeof(LUID) == sizeof(uint64_t), "LUID and uint64_t are not the same size");
      uint64_t key = GetLuidKey(desc.AdapterLuid);

      DeviceInfo& info = device_info[key];
      info.type = OrtHardwareDeviceType_GPU;
      info.vendor_id = desc.VendorId;
      info.device_id = desc.DeviceId;
      info.description = std::wstring(desc.Description);

      info.metadata[L"VideoMemory"] = std::to_wstring(desc.DedicatedVideoMemory / (1024 * 1024)) + L" MB";
      info.metadata[L"SystemMemory"] = std::to_wstring(desc.DedicatedSystemMemory / (1024 * 1024)) + L" MB";
      info.metadata[L"SharedSystemMemory"] = std::to_wstring(desc.DedicatedSystemMemory / (1024 * 1024)) + L" MB";
      info.metadata[L"HighPerformanceIndex"] = std::to_wstring(i);
    } while (false);

    adapter->Release();
  }

  /* TODO: Do we need to create a device to discover meaningful info?
    ID3D12Device* device = nullptr;
    hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
      std::cerr << "Failed to create D3D12 device.\n";
      adapter->Release();
      factory->Release();
      return -1;
    }

    // Check for compute shader support (feature level 11_0+ implies basic compute support)
    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
    if (SUCCEEDED(hr)) {
      std::cout << "Compute shader supported via D3D12." << std::endl;
      std::cout << "Tiled resources tier: " << options.TiledResourcesTier << std::endl;
      std::cout << "Resource binding tier: " << options.ResourceBindingTier << std::endl;
    }

    device->Release();
    */
  factory->Release();

  return device_info;
}

#if DXCORE_AVAILABLE
// returns LUID to DeviceInfo
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoDxcore() {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  // Getting CPU info from SetupApi
  // Get information about the CPU device
  // auto vendor = CPUIDInfo::GetCPUIDInfo().GetCPUVendor();

  //// Create an OrtDevice for the CPU and add it to the list of found devices
  // OrtHardwareDevice cpu_device{OrtHardwareDeviceType_CPU, 0, std::string(vendor), 0};
  // device_info.insert({-1, cpu_device});

  // Get all GPUs and NPUs by querying WDDM/MCDM.
  wil::com_ptr<IDXCoreAdapterFactory> adapterFactory;
  THROW_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapterFactory)));

  // Look for devices that expose compute engines
  std::vector<const GUID*> allowedAttributes;
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);
  allowedAttributes.push_back(&DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU);

  // These attributes are not OR'd.  Have to query one at a time to get a full view.
  for (const auto& hwAttribute : allowedAttributes) {
    wil::com_ptr<IDXCoreAdapterList> adapterList;
    if (FAILED(adapterFactory->CreateAdapterList(1, hwAttribute, IID_PPV_ARGS(&adapterList)))) {
      continue;
    }

    const uint32_t adapterCount{adapterList->GetAdapterCount()};
    for (uint32_t adapterIndex = 0; adapterIndex < adapterCount; adapterIndex++) {
      wil::com_ptr<IDXCoreAdapter> adapter;
      THROW_IF_FAILED(adapterList->GetAdapter(adapterIndex, IID_PPV_ARGS(&adapter)));

      // Ignore software based devices
      bool isHardware{false};
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware)) || !isHardware) {
        continue;
      }

      HRESULT hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      static_assert(sizeof(LUID) == sizeof(uint64_t), "LUID and uint64_t are not the same size");
      LUID luid;  // really a LUID but we only need it to skip duplicated devices
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::InstanceLuid)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, sizeof(luid), &luid);
      } else {
        continue;
      }

      uint64_t key = GetLuidKey(luid);
      if (device_info.find(key) != device_info.end()) {
        // already found this device
        continue;
      }

      DeviceInfo& info = device_info[key];

      // Get hardware identifying information
      DXCoreHardwareIDParts idParts = {};
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareIDParts)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(idParts), &idParts);
        info.vendor_id = idParts.vendorID;
        info.device_id = idParts.deviceID;
      }

      // Is this a GPU or NPU
      if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
        info.type = OrtHardwareDeviceType::OrtHardwareDeviceType_GPU;
      } else {
        info.type = OrtHardwareDeviceType::OrtHardwareDeviceType_NPU;
      }

      // this returns char_t on US-EN Windows. assuming it returns wchar_t on other locales
      // the description from SetupApi is wchar_t to assuming we have that and don't need this one.
      // hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      // std::wstring driverDescription;
      // driverDescription.resize(256);
      //// this doesn't seem to return wchar_t
      // if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
      //   hrId = adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof(driverDescription),
      //                               &driverDescription);
      //   info.description = driverDescription;
      // }
    }
  }

  return device_info;
}
#endif

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_map<uint64_t, DeviceInfo> luid_to_dxinfo;  // dxcore info. key is luid
  std::unordered_set<uint64_t> npus;                        // NPU devices found in dxcore info
#if DXCORE_AVAILABLE
  luid_to_dxinfo = GetDeviceInfoDxcore();
  for (auto& [luid, device] : luid_to_dxinfo) {
    if (device.type == OrtHardwareDeviceType_NPU) {
      npus.insert(GetDeviceKey(device));
    }
  }
#endif

  // d3d12 info. key is luid
  std::unordered_map<uint64_t, DeviceInfo> luid_to_d3d12_info = GetDeviceInfoD3D12();
  // setupapi_info. key is vendor_id+device_id
  std::unordered_map<uint64_t, DeviceInfo> setupapi_info = GetDeviceInfoSetupApi(npus);

  // add dxcore info for any devices that are not in d3d12.
  // d3d12 info is more complete and has a good description and metadata
  for (auto& [luid, device] : luid_to_dxinfo) {
    if (luid_to_d3d12_info.find(luid) == luid_to_d3d12_info.end()) {
      luid_to_d3d12_info[luid] = device;
    }
  }

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;  // wstring to string
  const auto device_to_ortdevice = [&converter](
                                       DeviceInfo& device,
                                       std::unordered_map<std::wstring, std::wstring>* extra_metadata = nullptr) {
    OrtHardwareDevice ortdevice{device.type, device.vendor_id, device.device_id, converter.to_bytes(device.vendor)};

    if (device.bus_ids.size() > 0) {
      // use the first bus number. not sure how to handle multiple
      ortdevice.metadata.Add("BusNumber", std::to_string(device.bus_ids.back()).c_str());
      device.bus_ids.pop_back();
    }

    if (!device.description.empty()) {
      ortdevice.metadata.Add("Description", converter.to_bytes(device.description));
    }

    for (auto& [key, value] : device.metadata) {
      ortdevice.metadata.Add(converter.to_bytes(key), converter.to_bytes(value));
    }

    if (extra_metadata) {
      // add any extra metadata from the dxcore info
      for (auto& [key, value] : *extra_metadata) {
        if (device.metadata.find(key) == device.metadata.end()) {
          ortdevice.metadata.Add(converter.to_bytes(key), converter.to_bytes(value));
        }
      }
    }

    return ortdevice;
  };

  // create final set of devices with info from everything
  std::unordered_set<OrtHardwareDevice> devices;

  // CPU from SetupAPI
  for (auto& [idstr, device] : setupapi_info) {
    OrtHardwareDevice ort_device;
    if (device.type == OrtHardwareDeviceType_CPU) {
      // use the SetupApi info as-is
      devices.emplace(device_to_ortdevice(device));
    }
  }

  // filter GPU/NPU to devices in combined d3d12/dxcore info.
  for (auto& [luid, device] : luid_to_d3d12_info) {
    if (auto it = setupapi_info.find(GetDeviceKey(device)); it != setupapi_info.end()) {
      // use SetupApi info. merge metadata.
      devices.emplace(device_to_ortdevice(it->second, &device.metadata));
    } else {
      // no matching entry in SetupApi. use the dxinfo. no vendor. no BusNumber.
      devices.emplace(device_to_ortdevice(device));
    }
  }

  return devices;
}

}  // namespace onnxruntime
