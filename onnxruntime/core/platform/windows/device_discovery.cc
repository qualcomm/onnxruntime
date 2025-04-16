// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <unordered_set>

#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wil/com.h>

// GetDxgiInfo
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")
#include <iostream>

#include "core/common/cpuid_info.h"
#include "core/session/abi_devices.h"

// TEMPORARY: The CI builds target Windows 10 so do not have these GUIDs.
// This is to make the builds pass so any other issues can be resolved, but needs a real solution prior to checkin.
// these values were added in 10.0.22621.0 as part of DirectXCore API
#if NTDDI_VERSION < NTDDI_WIN10_RS5
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88);
DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, 0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed);
#endif

#include <Windows.h>
#include <SetupAPI.h>
#include <devguid.h>  // GUID_DEVCLASS_DISPLAY
#include <cfgmgr32.h>
#pragma comment(lib, "setupapi.lib")

#include <cassert>
namespace onnxruntime {

namespace {

// is the best we can do matching on description?
struct ExtraInfo {
  std::string vendor;
  std::string description;
  std::vector<DWORD> bus_ids;
};

std::unordered_map<std::string, ExtraInfo> GetExtraGpuInfo() {
  std::unordered_map<std::string, ExtraInfo> device_info;

  HDEVINFO devInfo = SetupDiGetClassDevs(&GUID_DEVCLASS_DISPLAY, nullptr, nullptr, DIGCF_PRESENT);
  if (devInfo == INVALID_HANDLE_VALUE) {
    return device_info;
  }

  SP_DEVINFO_DATA devData = {};
  devData.cbSize = sizeof(SP_DEVINFO_DATA);

  for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &devData); ++i) {
    CHAR buffer[1024];
    DWORD size = 0;

    ExtraInfo* entry = nullptr;

    // Get device instance ID (contains PCI path)
    if (CM_Get_Device_IDA(devData.DevInst, buffer, 1024, 0) == CR_SUCCESS) {
      // std::wcout << L"Device Instance: " << buffer << std::endl;
      // PCI\\VEN_xxxx&DEV_yyyy&...
      // Cut at the & after the 4 DEV hex chars
      std::string pcie_id = std::string(buffer).substr(0, 21);
      if (device_info.find(pcie_id) == device_info.end()) {
        device_info[pcie_id] = {};
      }

      entry = &device_info[pcie_id];
    } else {
      continue;
    }

    //// Get hardware ID (contains VEN_xxxx&DEV_xxxx)
    // if (SetupDiGetDeviceRegistryPropertyW(devInfo,
    //                                       &devData,
    //                                       SPDRP_HARDWAREID,
    //                                       nullptr,
    //                                       (PBYTE)buffer,
    //                                       sizeof(buffer),
    //                                       &size)) {
    //   std::wcout << L"Hardware ID: " << buffer << std::endl;
    // }

    // Get device description. Using SetupDiGetDeviceRegistryPropertyA as the OrtKeyValuePairs we're storing this in
    // as metadata use std::string based.
    if (SetupDiGetDeviceRegistryPropertyA(devInfo, &devData, SPDRP_DEVICEDESC, nullptr,
                                          (PBYTE)buffer, sizeof(buffer), &size)) {
      entry->description = &buffer[0];
    }

    DWORD busNumber = 0;
    size = 0;
    if (SetupDiGetDeviceRegistryPropertyA(devInfo, &devData, SPDRP_BUSNUMBER, nullptr,
                                          reinterpret_cast<PBYTE>(&busNumber), sizeof(busNumber), &size)) {
      // push_back in case there are two identical devices. not sure how else to tell them apart
      entry->bus_ids.push_back(busNumber);
    }

    DWORD regDataType = 0;
    if (SetupDiGetDeviceRegistryPropertyA(devInfo, &devData, SPDRP_MFG, &regDataType,
                                          (PBYTE)buffer, sizeof(buffer), &size)) {
      entry->vendor = &buffer[0];
    }
  }

  SetupDiDestroyDeviceInfoList(devInfo);

  return device_info;
}

std::unordered_map<int64_t, OrtHardwareDevice> GetInferencingDevices() {
  std::unordered_map<int64_t, OrtHardwareDevice> found_devices;

  // Get information about the CPU device
  auto vendor = CPUIDInfo::GetCPUIDInfo().GetCPUVendor();

  // Create an OrtDevice for the CPU and add it to the list of found devices
  OrtHardwareDevice cpu_device{OrtHardwareDeviceType_CPU, 0, std::string(vendor), 0};
  found_devices.insert({-1, cpu_device});

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
      int64_t luId = 0;  // really a LUID but we only need it to skip duplicated devices
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::InstanceLuid)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, sizeof(luId), &luId);
      } else {
        continue;
      }

      if (found_devices.find(luId) != found_devices.end()) {
        // already found this device
        continue;
      }

      // Get hardware identifying information
      DXCoreHardwareIDParts idParts = {};
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareIDParts)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(idParts), &idParts);
      }

      std::string_view mcdm_vendor_managed;
      if (SUCCEEDED(hrId) && adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareID)) {
        DXCoreHardwareID id;
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(id), &id);
        if (SUCCEEDED(hrId)) {
          idParts.vendorID = id.vendorID;
          idParts.deviceID = id.deviceID;
          idParts.revisionID = id.revision;
          idParts.subSystemID = id.subSysID;
          idParts.subVendorID = 0;
        }
      }

      // Is this a GPU or NPU
      OrtHardwareDeviceType kind = OrtHardwareDeviceType::OrtHardwareDeviceType_NPU;
      if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
        kind = OrtHardwareDeviceType::OrtHardwareDeviceType_GPU;
      }

      OrtHardwareDevice found_device(kind, idParts.vendorID, "", idParts.deviceID);

      // TODO: Get hardware properties given these ID parts and decide what to do with them
      hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      char driverDescription[256];
      // technically this returns a wchar_t but the key/values in OrtKeyValuePairs are char*.
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof(driverDescription),
                                    &driverDescription);
        found_device.metadata.Add("Description", driverDescription);
      }

      // Insert the device into the set - if not a duplicate
      found_devices.insert({luId, found_device});
    }
  }

  return found_devices;
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_map<int64_t, OrtHardwareDevice> devices = GetInferencingDevices();

  std::vector<std::pair<std::string, OrtHardwareDevice*>> pcie_id_and_devices;
  pcie_id_and_devices.reserve(devices.size());

  for (auto& [luid, device] : devices) {
    std::stringstream ss;
    ss << std::uppercase << std::hex << "PCI\\VEN_" << device.vendor_id << "&DEV_" << device.bus_id;
    std::string pcie_id = ss.str();
    pcie_id_and_devices.emplace_back(pcie_id, &device);
  }

  auto extra_info = GetExtraGpuInfo();
  for (auto& [pcie_id, device] : pcie_id_and_devices) {
    if (auto it = extra_info.find(pcie_id); it != extra_info.end()) {
      auto& info = it->second;
      assert(info.description == device->metadata.entries["Description"]);

      device->vendor = info.vendor;
      // remove one bus_id from the list in case there are multiple.
      device->metadata.Add("BusNumber", std::to_string(info.bus_ids.back()).c_str());
      info.bus_ids.pop_back();
    }
  }

  // now that all the values are final create the set.
  std::unordered_set<OrtHardwareDevice> final_devices;
  for (auto& [luid, device] : devices) {
    final_devices.emplace(std::move(device));
  }

  return final_devices;
}

}  // namespace onnxruntime
