// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang texpands ST — case-table driven.
// Each case launches a different kernel variant, writes from per-case subdirectory.
// Numerical comparison is done externally by compare.py.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// Kernel launch wrappers (defined in launch.cpp)
void LaunchTEXPANDS_f32_16x64_scalar5(float *dst, void *stream);
void LaunchTEXPANDS_f32_32x32_scalar3(float *dst, void *stream);
void LaunchTEXPANDS_f32_16x64_partial(float *dst, void *stream);
void LaunchTEXPANDS_i32_64x64_scalar100(int32_t *dst, void *stream);
void LaunchTEXPANDS_f16_64x64_scalar1_5(uint16_t *dst, void *stream);
void LaunchTEXPANDS_i16_64x64_scalar50(int16_t *dst, void *stream);

enum class DataType { F32, I32, F16, I16 };

struct TestCase {
    const char *name;
    DataType    dtype;
    void (*launch)(void *, void *);  // Generic launch function pointer
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      elemSize;   // bytes per element
};

// Helper to wrap type-specific launch functions
template<typename T>
void wrapLaunch(void *dst, void *stream, void (*fn)(T *, void *)) {
    fn((T *)dst, stream);
}

static const TestCase kCases[] = {
    // ========== float32 cases ==========
{"f32_16x64_scalar5", DataType::F32,
     [](void *dst, void *stream) { wrapLaunch<float>(dst, stream, LaunchTEXPANDS_f32_16x64_scalar5); },
     16, 64, 16, 64, sizeof(float)},
{"f32_16x64_partial", DataType::F32,
     [](void *dst, void *stream) { wrapLaunch<float>(dst, stream, LaunchTEXPANDS_f32_16x64_partial); },
     16, 64, 12, 48, sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t fileSize  = elemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    // Per-case data directory
    std::string caseDir = std::string("./") + tc.name;

    void *dstHost = nullptr;
    void *dstDevice = nullptr;

    aclrtMallocHost(&dstHost, fileSize);
    aclrtMalloc(&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Launch kernel (scalar is hardcoded in .pto)
    tc.launch(dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (dstHost != nullptr)
        aclrtFreeHost(dstHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    // Optional case filter: ./texpands [case_name]
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}
