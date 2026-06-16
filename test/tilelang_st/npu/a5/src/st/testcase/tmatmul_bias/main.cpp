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
void LaunchTMATMUL_BIAS_f16_16x16x16(void *a, void *b, void *bias, void *c, void *stream);
void LaunchTMATMUL_BIAS_f16_bias_f16_16x15x16(void *a, void *b, void *bias, void *c, void *stream);
void LaunchTMATMUL_BIAS_f16_bias_bf16_112x127x80(void *a, void *b, void *bias, void *c, void *stream);
void LaunchTMATMUL_BIAS_bf16_bias_bf16_80x112x63(void *a, void *b, void *bias, void *c, void *stream);
void LaunchTMATMUL_BIAS_f32_bias_f32_127x128x63(void *a, void *b, void *bias, void *c, void *stream);

void LaunchTMATMUL_BIAS_i8_bias_i32_8x7x6(void *a, void *b, void *bias, void *c, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      M;         // valid rows
    size_t      K;
    size_t      N;         // valid cols
    size_t      M_aligned; // aligned rows (tileM)
    size_t      K_aligned; // aligned inner dim (tileK)
    size_t      N_aligned; // aligned cols (tileN)
    size_t      aElemSize;
    size_t      bElemSize;
    size_t      biasElemSize;
    size_t      cElemSize;
};

static const TestCase kCases[] = {
    {"f16_16x16x16",                         LaunchTMATMUL_BIAS_f16_16x16x16,                         16,  16,  16,  16,  16,  16,  2, 2, 4, 4},

    {"i8_bias_i32_8x7x6",                   LaunchTMATMUL_BIAS_i8_bias_i32_8x7x6,              8,   7,   6,   16,  32,  32,  1, 1, 4, 4},

    {"f16_bias_f16_16x15x16",              LaunchTMATMUL_BIAS_f16_bias_f16_16x15x16,             16,  15,  16,  16,  16,  16,  2, 2, 4, 4},  // DEBUG: f32 bias
    {"f16_bias_bf16_112x127x80",           LaunchTMATMUL_BIAS_f16_bias_bf16_112x127x80,         112, 127, 80,  112, 128, 80,  2, 2, 4, 4},
    {"bf16_bias_bf16_80x112x63",           LaunchTMATMUL_BIAS_bf16_bias_bf16_80x112x63,          80, 112, 63,  80,  128, 64,  2, 2, 4, 4},
    {"f32_bias_f32_127x128x63",            LaunchTMATMUL_BIAS_f32_bias_f32_127x128x63,          127, 128, 63,  128, 128, 64,  4, 4, 4, 4},

};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    // Allocate device buffers at aligned sizes so GM→L1 loads don't read OOB.
    const size_t aBytes = tc.M_aligned * tc.K_aligned * tc.aElemSize;
    const size_t bBytes = tc.K_aligned * tc.N_aligned * tc.bElemSize;
    const size_t biasBytes = tc.N_aligned * tc.biasElemSize;
    const size_t cBytes = tc.M_aligned * tc.N_aligned * tc.cElemSize;
    size_t aFileSize = aBytes;
    size_t bFileSize = bBytes;
    size_t biasFileSize = biasBytes;

    std::printf(
        "[INFO] === case: %s (M=%zu, K=%zu, N=%zu, M_aligned=%zu, N_aligned=%zu) ===\n",
        tc.name, tc.M, tc.K, tc.N, tc.M_aligned, tc.N_aligned
    );

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr, *bHost = nullptr, *biasHost = nullptr, *cHost = nullptr;
    void *aDevice = nullptr, *bDevice = nullptr, *biasDevice = nullptr, *cDevice = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&biasHost, biasBytes);
    aclrtMallocHost(&cHost, cBytes);

    aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&biasDevice, biasBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&cDevice, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aFileSize, aHost, aBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bFileSize, bHost, bBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input3.bin").c_str(), biasFileSize, biasHost, biasBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(biasDevice, biasBytes, biasHost, biasBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(aDevice, bDevice, biasDevice, cDevice, stream);

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
