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
namespace onnxruntime {

namespace {
int PrintDxgiInfo() {
  IDXGIFactory* pFactory = nullptr;
  HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory));

  if (FAILED(hr)) {
    std::cerr << "Failed to create DXGI Factory!" << std::endl;
    return 1;
  }

  IDXGIAdapter* pAdapter = nullptr;
  for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC desc;
    pAdapter->GetDesc(&desc);

    std::wcout << L"GPU: " << desc.Description << std::endl;
    std::wcout << L"Vendor ID: " << desc.VendorId << std::endl;
    std::wcout << L"Device ID: " << desc.DeviceId << std::endl;
    std::wcout << L"SubSys ID: " << desc.SubSysId << std::endl;
    std::wcout << L"Revision: " << desc.Revision << std::endl;
    std::wcout << L"-----------------------------------" << std::endl;

    pAdapter->Release();
  }

  pFactory->Release();
  return 0;
}

std::vector<HardwareDevice> GetGpuAndNpuDevices() {
  std::vector<HardwareDevice> found_devices;
  std::unordered_set<uint32_t> found_device_ids;

  // Get information about the CPU device

  // Get all GPUs and NPUs by querying WDDM/MCDM.
  wil::com_ptr<IDXCoreAdapterFactory> adapterFactory;
  THROW_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapterFactory)));

  // Look for devices that expose compute engines
  std::vector<const GUID*> allowedAttributes;
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);
  allowedAttributes.push_back(&DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU);
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
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware)) {
        continue;
      }
      bool isHardware{false};
      if (FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware)) || !isHardware) {
        continue;
      }

      // Get hardware identifying information
      DXCoreHardwareIDParts idParts = {};
      HRESULT hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareIDParts)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(idParts), &idParts);
      }

      if (FAILED(hrId) && adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareID)) {
        DXCoreHardwareID id;
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(id), &id);
        if (SUCCEEDED(hrId)) {
          idParts.vendorID = id.vendorID;
          idParts.deviceID = id.deviceID;
          idParts.revisionID = id.revision;
          idParts.subSystemID = id.subSysID;
        }
      }

      if (found_device_ids.count(idParts.deviceID) > 0) {
        continue;  // already found this device
      }

      // TODO: Get hardware properties given these ID parts
      //
      // TODO: Should OrtDevice really be int not uint and 16 not 32
      // OrtDevice found_device(kind, OrtDevice::MemType::DEFAULT, (int16_t)idParts.deviceID, (int16_t)idParts.vendorID);
      // found_devices.push_back(found_device);

      // Is this a GPU or NPU
      // TODO: Why wouldn't checking for DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU first discover all NPUs?
      //       If so wouldn't any later matches be skipped as the device is already in found_devices_ids?
      if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
        // GPU
        HardwareDevice device;
        device.type = HardwareDevice::Type::GPU;
        device.vendor = std::to_string(idParts.vendorID);

      } else {
        // NPU.
        //
      }

      found_device_ids.insert(idParts.deviceID);
    }
  }

  return found_devices;
}
}  // namespace

std::vector<HardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::vector<HardwareDevice> devices;
  // get CPU devices

  // get GPU devices

  // get NPU devices

  PrintDxgiInfo();
  devices = GetGpuAndNpuDevices();
  return devices;
}

}  // namespace onnxruntime
