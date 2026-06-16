// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tmrgsort ST — case-table driven.
// Each case launches a different kernel variant, reads/writes from per-case subdirectory.
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
void LaunchTMRGSORT_f32_single_1x256_b64(float *src, float *dst, void *stream);
void LaunchTMRGSORT_f32_single_1x320_b64(float *src, float *dst, void *stream);
void LaunchTMRGSORT_f32_single_1x640_b64(float *src, float *dst, void *stream);
void LaunchTMRGSORT_f16_single_1x320_b64(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTMRGSORT_f16_single_1x1024_b256(uint16_t *src, uint16_t *dst, void *stream);

// Multi-list launch wrappers
void LaunchTMRGSORT_f16_2list_b64_basic(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTMRGSORT_f32_3list_b64_basic(float *src0, float *src1, float *src2, float *dst, void *stream);
void LaunchTMRGSORT_f16_3list_exhausted(uint16_t *src0, uint16_t *src1, uint16_t *src2, uint16_t *dst, void *stream);
void LaunchTMRGSORT_f32_4list_non_uniform(float *src0, float *src1, float *src2, float *src3, float *dst, void *stream);
void LaunchTMRGSORT_f16_4list_basic(uint16_t *src0, uint16_t *src1, uint16_t *src2, uint16_t *src3, uint16_t *dst, void *stream);

// TopK launch wrappers
void LaunchTMRGSORT_f32_topk_2048_2048(float *src, float *dst, void *stream);
void LaunchTMRGSORT_f16_topk_2048_1024(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTMRGSORT_f16_topk_1280_512(uint16_t *src, uint16_t *dst, void *stream);

using LaunchFn = void (*)(void *, void *, void *);
using LaunchFn2 = void (*)(void *, void *, void *, void *);
using LaunchFn3 = void (*)(void *, void *, void *, void *, void *);
using LaunchFn4 = void (*)(void *, void *, void *, void *, void *, void *);

struct TestCase {
    const char *name;
    int         listNum;    // 1 for single-list, 2/3/4 for multi-list
    LaunchFn    launch;     // for single-list
    LaunchFn2   launch2;    // for 2-list
    LaunchFn3   launch3;    // for 3-list
    LaunchFn4   launch4;    // for 4-list
    size_t      srcRows;
    size_t      srcCols;    // for single-list: element count
    size_t      srcCols0;   // for multi-list: src0 element count
    size_t      srcCols1;   // for multi-list: src1 element count
    size_t      srcCols2;   // for multi-list: src2 element count (for 3/4-list)
    size_t      srcCols3;   // for multi-list: src3 element count (for 4-list)
    size_t      dstRows;
    size_t      dstCols;    // element count
    size_t      elemSize;    // bytes per element (4 for f32, 2 for f16)
    size_t      structSize;  // 8 bytes per (value, index) pair
    size_t      elemsPerStruct; // structSize / elemSize (2 for f32, 4 for f16)
};

static const TestCase kCases[] = {
    // Single-list cases (Format1)
{"f32_single_1x256_b64",   1, reinterpret_cast<LaunchFn>(LaunchTMRGSORT_f32_single_1x256_b64),   nullptr, nullptr, nullptr, 1, 256,  0, 0, 0, 0, 1, 256,  sizeof(float),    8, 2},
{"f16_topk_1280_512",   1, reinterpret_cast<LaunchFn>(LaunchTMRGSORT_f16_topk_1280_512),   nullptr, nullptr, nullptr, 1, 1280, 0, 0, 0, 0, 1,  512, sizeof(uint16_t), 8, 4},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    int rc = 0;
    std::string caseDir = std::string("./") + tc.name;

    // Single-list case (Format1)
    if (tc.listNum == 1) {
        // srcCols/dstCols are in ELEMENTS, need to convert to STRUCTURE count
        // elemsPerStruct = structSize / elemSize (2 for f32, 4 for f16)
        size_t srcStructs = tc.srcCols / tc.elemsPerStruct;
        size_t dstStructs = tc.dstCols / tc.elemsPerStruct;

        // File sizes in bytes
        size_t srcFileSize = tc.srcRows * srcStructs * tc.structSize;
        size_t dstFileSize = tc.dstRows * dstStructs * tc.structSize;

        std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu) ===\n",
                    tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols);

        void *srcHost = nullptr, *dstHost = nullptr;
        void *srcDevice = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&srcHost), srcFileSize);
        aclrtMallocHost((void **)(&dstHost), dstFileSize);

        aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (!ReadFile((caseDir + "/input0.bin").c_str(), srcFileSize, srcHost, srcFileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launch(srcDevice, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (srcDevice != nullptr)
            aclrtFree(srcDevice);
        if (dstDevice != nullptr)
            aclrtFree(dstDevice);
        if (srcHost != nullptr)
            aclrtFreeHost(srcHost);
        if (dstHost != nullptr)
            aclrtFreeHost(dstHost);

        if (rc == 0)
            std::printf("[INFO] case %s done\n", tc.name);
    }

    // Multi-list case (Format2)
    else if (tc.listNum == 2) {
        // For 2-list: src0, src1, dst
        // srcCols0, srcCols1 are in ELEMENTS, dstCols in ELEMENTS
        // elemsPerStruct = structSize / elemSize (2 for f32, 4 for f16)
        size_t src0Structs = tc.srcCols0 / tc.elemsPerStruct;
        size_t src1Structs = tc.srcCols1 / tc.elemsPerStruct;
        size_t dstStructs = tc.dstCols / tc.elemsPerStruct;

        size_t src0FileSize = tc.srcRows * src0Structs * tc.structSize;
        size_t src1FileSize = tc.srcRows * src1Structs * tc.structSize;
        size_t dstFileSize = tc.dstRows * dstStructs * tc.structSize;

        std::printf("[INFO] === case: %s (src0=%zux%zu, src1=%zux%zu, dst=%zux%zu) ===\n",
                    tc.name, tc.srcRows, tc.srcCols0, tc.srcRows, tc.srcCols1, tc.dstRows, tc.dstCols);

        void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
        void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&src0Host), src0FileSize);
        aclrtMallocHost((void **)(&src1Host), src1FileSize);
        aclrtMallocHost((void **)(&dstHost), dstFileSize);

        aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        // Read input0.bin and input1.bin
        if (!ReadFile((caseDir + "/input0.bin").c_str(), src0FileSize, src0Host, src0FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input1.bin").c_str(), src1FileSize, src1Host, src1FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launch2(src0Device, src1Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (src0Device != nullptr) aclrtFree(src0Device);
        if (src1Device != nullptr) aclrtFree(src1Device);
        if (dstDevice != nullptr)  aclrtFree(dstDevice);
        if (src0Host != nullptr)   aclrtFreeHost(src0Host);
        if (src1Host != nullptr)   aclrtFreeHost(src1Host);
        if (dstHost != nullptr)    aclrtFreeHost(dstHost);

        if (rc == 0)
            std::printf("[INFO] case %s done\n", tc.name);
    }

    // 3-list case (Format3)
    else if (tc.listNum == 3) {
        size_t src0Structs = tc.srcCols0 / tc.elemsPerStruct;
        size_t src1Structs = tc.srcCols1 / tc.elemsPerStruct;
        size_t src2Structs = tc.srcCols2 / tc.elemsPerStruct;
        size_t dstStructs = tc.dstCols / tc.elemsPerStruct;

        size_t src0FileSize = tc.srcRows * src0Structs * tc.structSize;
        size_t src1FileSize = tc.srcRows * src1Structs * tc.structSize;
        size_t src2FileSize = tc.srcRows * src2Structs * tc.structSize;
        size_t dstFileSize = tc.dstRows * dstStructs * tc.structSize;

        std::printf("[INFO] === case: %s (src0=%zux%zu, src1=%zux%zu, src2=%zux%zu, dst=%zux%zu) ===\n",
                    tc.name, tc.srcRows, tc.srcCols0, tc.srcRows, tc.srcCols1,
                    tc.srcRows, tc.srcCols2, tc.dstRows, tc.dstCols);

        void *src0Host = nullptr, *src1Host = nullptr, *src2Host = nullptr, *dstHost = nullptr;
        void *src0Device = nullptr, *src1Device = nullptr, *src2Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&src0Host), src0FileSize);
        aclrtMallocHost((void **)(&src1Host), src1FileSize);
        aclrtMallocHost((void **)(&src2Host), src2FileSize);
        aclrtMallocHost((void **)(&dstHost), dstFileSize);

        aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src2Device, src2FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (!ReadFile((caseDir + "/input0.bin").c_str(), src0FileSize, src0Host, src0FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input1.bin").c_str(), src1FileSize, src1Host, src1FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), src2FileSize, src2Host, src2FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src2Device, src2FileSize, src2Host, src2FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launch3(src0Device, src1Device, src2Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (src0Device != nullptr) aclrtFree(src0Device);
        if (src1Device != nullptr) aclrtFree(src1Device);
        if (src2Device != nullptr) aclrtFree(src2Device);
        if (dstDevice != nullptr)  aclrtFree(dstDevice);
        if (src0Host != nullptr)   aclrtFreeHost(src0Host);
        if (src1Host != nullptr)   aclrtFreeHost(src1Host);
        if (src2Host != nullptr)   aclrtFreeHost(src2Host);
        if (dstHost != nullptr)    aclrtFreeHost(dstHost);

        if (rc == 0)
            std::printf("[INFO] case %s done\n", tc.name);
    }

    // 4-list case (Format4)
    else if (tc.listNum == 4) {
        size_t src0Structs = tc.srcCols0 / tc.elemsPerStruct;
        size_t src1Structs = tc.srcCols1 / tc.elemsPerStruct;
        size_t src2Structs = tc.srcCols2 / tc.elemsPerStruct;
        size_t src3Structs = tc.srcCols3 / tc.elemsPerStruct;
        size_t dstStructs = tc.dstCols / tc.elemsPerStruct;

        size_t src0FileSize = tc.srcRows * src0Structs * tc.structSize;
        size_t src1FileSize = tc.srcRows * src1Structs * tc.structSize;
        size_t src2FileSize = tc.srcRows * src2Structs * tc.structSize;
        size_t src3FileSize = tc.srcRows * src3Structs * tc.structSize;
        size_t dstFileSize = tc.dstRows * dstStructs * tc.structSize;

        std::printf("[INFO] === case: %s (src0=%zux%zu, src1=%zux%zu, src2=%zux%zu, src3=%zux%zu, dst=%zux%zu) ===\n",
                    tc.name, tc.srcRows, tc.srcCols0, tc.srcRows, tc.srcCols1,
                    tc.srcRows, tc.srcCols2, tc.srcRows, tc.srcCols3,
                    tc.dstRows, tc.dstCols);

        void *src0Host = nullptr, *src1Host = nullptr, *src2Host = nullptr, *src3Host = nullptr, *dstHost = nullptr;
        void *src0Device = nullptr, *src1Device = nullptr, *src2Device = nullptr, *src3Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&src0Host), src0FileSize);
        aclrtMallocHost((void **)(&src1Host), src1FileSize);
        aclrtMallocHost((void **)(&src2Host), src2FileSize);
        aclrtMallocHost((void **)(&src3Host), src3FileSize);
        aclrtMallocHost((void **)(&dstHost), dstFileSize);

        aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src2Device, src2FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src3Device, src3FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (!ReadFile((caseDir + "/input0.bin").c_str(), src0FileSize, src0Host, src0FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input1.bin").c_str(), src1FileSize, src1Host, src1FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), src2FileSize, src2Host, src2FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input3.bin").c_str(), src3FileSize, src3Host, src3FileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src2Device, src2FileSize, src2Host, src2FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src3Device, src3FileSize, src3Host, src3FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launch4(src0Device, src1Device, src2Device, src3Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (src0Device != nullptr) aclrtFree(src0Device);
        if (src1Device != nullptr) aclrtFree(src1Device);
        if (src2Device != nullptr) aclrtFree(src2Device);
        if (src3Device != nullptr) aclrtFree(src3Device);
        if (dstDevice != nullptr)  aclrtFree(dstDevice);
        if (src0Host != nullptr)   aclrtFreeHost(src0Host);
        if (src1Host != nullptr)   aclrtFreeHost(src1Host);
        if (src2Host != nullptr)   aclrtFreeHost(src2Host);
        if (src3Host != nullptr)   aclrtFreeHost(src3Host);
        if (dstHost != nullptr)    aclrtFreeHost(dstHost);

        if (rc == 0)
            std::printf("[INFO] case %s done\n", tc.name);
    }

    else {
        std::fprintf(stderr, "[ERROR] Unsupported listNum=%d for case %s\n", tc.listNum, tc.name);
        rc = 1;
    }

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
        int ret = RunCase(kCases[i], stream);
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
