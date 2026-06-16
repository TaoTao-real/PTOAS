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
void LaunchTGEMV_f16_1x300x60(void *a, void *b, void *c, void *stream);
void LaunchTGEMV_BIAS_f16_1x512x85(void *a, void *b, void *bias, void *c, void *stream);

using LaunchFn3 = void (*)(void *, void *, void *, void *);
using LaunchFn4 = void (*)(void *, void *, void *, void *, void *);

struct TestCase {
    const char *name;
    bool        hasBias;
    LaunchFn3   launch3;
    LaunchFn4   launch4;
    size_t      M;
    size_t      K;
    size_t      N;
    size_t      aElemSize;
    size_t      bElemSize;
    size_t      biasElemSize;
    size_t      cElemSize;
};

static const TestCase kCases[] = {
    {"gemv_f16_1x300x60",       false, LaunchTGEMV_f16_1x300x60,       nullptr,   1, 320,  64, 2, 2, 0, 4},
    {"gemv_bias_f16_1x512x85",  true,  nullptr,                       LaunchTGEMV_BIAS_f16_1x512x85,   1, 512,  96, 2, 2, 4, 4},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    size_t aBytes = tc.M * tc.K * tc.aElemSize;
    size_t bBytes = tc.K * tc.N * tc.bElemSize;
    size_t biasBytes = tc.hasBias ? tc.N * tc.biasElemSize : 0;
    const size_t cBytes = tc.M * tc.N * tc.cElemSize;

    std::printf(
        "[INFO] === case: %s (M=%zu, K=%zu, N=%zu, a_esize=%zu, b_esize=%zu, c_esize=%zu) ===\n",
        tc.name, tc.M, tc.K, tc.N, tc.aElemSize, tc.bElemSize, tc.cElemSize
    );

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr, *bHost = nullptr, *biasHost = nullptr, *cHost = nullptr;
    void *aDevice = nullptr, *bDevice = nullptr, *biasDevice = nullptr, *cDevice = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&cHost, cBytes);
    if (tc.hasBias) aclrtMallocHost(&biasHost, biasBytes);

    aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&cDevice, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (tc.hasBias) aclrtMalloc(&biasDevice, biasBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aBytes, aHost, aBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bBytes, bHost, bBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && tc.hasBias && !ReadFile((caseDir + "/input3.bin").c_str(), biasBytes, biasHost, biasBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (tc.hasBias) aclrtMemcpy(biasDevice, biasBytes, biasHost, biasBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        if (tc.hasBias) {
            tc.launch4(aDevice, bDevice, biasDevice, cDevice, stream);
        } else {
            tc.launch3(aDevice, bDevice, cDevice, stream);
        }

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(cHost, cBytes, cDevice, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), cHost, cBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (aDevice != nullptr) aclrtFree(aDevice);
    if (bDevice != nullptr) aclrtFree(bDevice);
    if (biasDevice != nullptr) aclrtFree(biasDevice);
    if (cDevice != nullptr) aclrtFree(cDevice);
    if (aHost != nullptr) aclrtFreeHost(aHost);
    if (bHost != nullptr) aclrtFreeHost(bHost);
    if (biasHost != nullptr) aclrtFreeHost(biasHost);
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
