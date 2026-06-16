// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

// ---- launch wrappers (defined in launch.cpp) ----
void LaunchTMATMUL_f16_40x50x60(void *a, void *b, void *c, void *stream);
void LaunchTMATMUL_i8_6x7x8(void *a, void *b, void *c, void *stream);
void LaunchTMATMUL_f16_127x128x61(void *a, void *b, void *c, void *stream);
void LaunchTMATMUL_f32_120x110x50(void *a, void *b, void *c, void *stream);
void LaunchTMATMUL_bf16_144x80x48(void *a, void *b, void *c, void *stream);
// void LaunchTMATMUL_f8e4m3_32x64x96(void *a, void *b, void *c, void *stream);
// void LaunchTMATMUL_f8e4m3_f8e5m2_128x96x64(void *a, void *b, void *c, void *stream);
// void LaunchTMATMUL_f8e5m2_f8e4m3_145x115x85(void *a, void *b, void *c, void *stream);
// void LaunchTMATMUL_f8e5m2_120x90x160(void *a, void *b, void *c, void *stream);
// void LaunchTMATMUL_hif8_30x90x60(void *a, void *b, void *c, void *stream);
void LaunchTMATMUL_f32_16x32x64(void *a, void *b, void *c, void *stream);
void LaunchTMATMUL_f32_128x96x64(void *a, void *b, void *c, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      M;
    size_t      K;
    size_t      N;
    size_t      aElemSize;
    size_t      bElemSize;
    size_t      cElemSize;
};

static const TestCase kCases[] = {
    // M/K/N values match gen_data padding: K_use is K rounded up to block size.
    {"f16_40x50x60",                    LaunchTMATMUL_f16_40x50x60,                    48,  64,  64,  2, 2, 4},
    {"i8_6x7x8",                        LaunchTMATMUL_i8_6x7x8,                        16,  32,  32,  1, 1, 4},
    {"f16_127x128x61",                  LaunchTMATMUL_f16_127x128x61,                 128, 128,  64,  2, 2, 4},
    {"f32_120x110x50",                  LaunchTMATMUL_f32_120x110x50,                 128, 112,  64,  4, 4, 4},
    {"bf16_144x80x48",                  LaunchTMATMUL_bf16_144x80x48,                 144,  80,  48,  2, 2, 4},
    // ...
    {"f32_16x32x64",                    LaunchTMATMUL_f32_16x32x64,                   16,  32,  64,  4, 4, 4},
    {"f32_128x96x64",                   LaunchTMATMUL_f32_128x96x64,                 128,  96,  64,  4, 4, 4},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    size_t aBytes = tc.M * tc.K * tc.aElemSize;
    size_t bBytes = tc.K * tc.N * tc.bElemSize;
    const size_t cBytes = tc.M * tc.N * tc.cElemSize;

    std::printf(
        "[INFO] === case: %s (M=%zu, K=%zu, N=%zu, a_esize=%zu, b_esize=%zu, c_esize=%zu) ===\n",
        tc.name, tc.M, tc.K, tc.N, tc.aElemSize, tc.bElemSize, tc.cElemSize
    );

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr, *bHost = nullptr, *cHost = nullptr;
    void *aDevice = nullptr, *bDevice = nullptr, *cDevice = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&cHost, cBytes);

    aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&cDevice, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aBytes, aHost, aBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bBytes, bHost, bBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(aDevice, bDevice, cDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(cHost, cBytes, cDevice, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), cHost, cBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (aDevice != nullptr) aclrtFree(aDevice);
    if (bDevice != nullptr) aclrtFree(bDevice);
    if (cDevice != nullptr) aclrtFree(cDevice);
    if (aHost != nullptr) aclrtFreeHost(aHost);
    if (bHost != nullptr) aclrtFreeHost(bHost);
    if (cHost != nullptr) aclrtFreeHost(cHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
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
