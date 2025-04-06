// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
// struct to provider ownership via std::string as well as support the GetKeyValuePairs
// TODO: Validate adding entries doesn't invalidate existing pointers. assuming std::unordered_map is smart enough to
//       std::move any strings in it.

struct OrtKeyValuePairs {
  std::unordered_map<std::string, std::string> entries;
  // members to make returning all key/value entries via the C API easier
  std::vector<const char*> keys;
  std::vector<const char*> values;

  void Copy(const std::unordered_map<std::string, std::string>& src) {
    entries = src;
    Sync();
  }

  void Add(const char* key, const char* value) {
    std::string key_str(key);
    auto iter_inserted = entries.insert({std::move(key_str), std::string(value)});
    bool inserted = iter_inserted.second;
    if (inserted) {
      const auto& entry = *iter_inserted.first;
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    } else {
      // rebuild is easier and this is not expected to be a common case. otherwise we need to to strcmp on all entries.
      Sync();
    }
  }

 private:
  void Sync() {
    keys.clear();
    values.clear();
    for (const auto& entry : entries) {
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    }
  }
};
