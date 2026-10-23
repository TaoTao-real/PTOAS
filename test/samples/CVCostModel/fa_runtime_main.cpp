// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef KERNEL_SYMBOL
#error KERNEL_SYMBOL must name the experiment-specific kernel
#endif
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void KERNEL_SYMBOL(
    uint32_t, uint32_t, void *, void *, void *, void *, void *, int64_t, int64_t);

void check(aclError error, const char *action) {
  if (error != ACL_SUCCESS)
    throw std::runtime_error(std::string(action) + " failed: " + std::to_string(error));
}

// Inputs are zero or exact binary fractions with magnitude <= 1.
uint16_t half(float value) {
  if (value == 0) return 0;
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return ((bits >> 16) & 0x8000) | ((((bits >> 23) & 255) - 112) << 10) |
         ((bits >> 13) & 1023);
}

void dump(const std::filesystem::path &path, const void *data, size_t bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char *>(data), bytes);
  if (!file) throw std::runtime_error("Cannot write output file");
}

struct Buffer {
  static constexpr size_t guard = 256;
  void *base = nullptr;
  size_t bytes;
  std::vector<uint8_t> initial;
  explicit Buffer(const void *data, size_t count)
      : bytes(count), initial(count + 2 * guard, 0xA5) {
    std::memcpy(initial.data() + guard, data, count);
    check(aclrtMalloc(&base, initial.size(), ACL_MEM_MALLOC_HUGE_FIRST), "Allocate buffer");
    check(aclrtMemcpy(base, initial.size(), initial.data(), initial.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE), "Restore buffer");
  }
  Buffer(const Buffer &) = delete;
  ~Buffer() { if (base) aclrtFree(base); }
  void *payload() const { return static_cast<uint8_t *>(base) + guard; }
  std::vector<uint8_t> read(bool input) const {
    std::vector<uint8_t> actual(initial.size());
    check(aclrtMemcpy(actual.data(), actual.size(), base, actual.size(),
                      ACL_MEMCPY_DEVICE_TO_HOST), "Read buffer");
    if (!std::equal(actual.begin(), actual.begin() + guard, initial.begin()) ||
        !std::equal(actual.end() - guard, actual.end(), initial.end() - guard))
      throw std::runtime_error("GM guard changed");
    if (input && actual != initial) throw std::runtime_error("Input changed");
    return actual;
  }
};

int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::runtime_error("Use: runner DEVICE CASE SEED NEW_OUTPUT_DIR");
    const int device = std::stoi(argv[1]);
    const std::string mode = argv[2];
    uint32_t state = static_cast<uint32_t>(std::stoul(argv[3]));
    const std::filesystem::path output = argv[4];
    if (!std::filesystem::create_directory(output))
      throw std::runtime_error("Output directory must be new");
    if (mode != "zero" && mode != "one_v" && mode != "one_qkv" && mode != "pattern")
      throw std::runtime_error("Unknown input case");
    constexpr int S1 = 128, S2 = 1024, D = 128;
    std::vector<float> qf(S1 * D, mode == "one_qkv" ? 1 : 0);
    std::vector<float> kf(S2 * D, mode == "one_qkv" ? 1 : 0);
    std::vector<float> vf(S2 * D, mode == "one_v" || mode == "one_qkv" ? 1 : 0);
    auto next = [&]() {
      state = state * 1664525u + 1013904223u;
      return static_cast<float>(static_cast<int>((state >> 16) % 17) - 8) / 16;
    };
    if (mode == "pattern") {
      for (auto &x : qf) x = next();
      for (auto &x : kf) x = next();
      for (auto &x : vf) x = next();
    }
    std::vector<uint16_t> q(qf.size()), k(kf.size()), v(vf.size());
    std::transform(qf.begin(), qf.end(), q.begin(), half);
    std::transform(kf.begin(), kf.end(), k.begin(), half);
    std::transform(vf.begin(), vf.end(), v.begin(), half);
    dump(output / "Q.f16", q.data(), q.size() * 2);
    dump(output / "K.f16", k.data(), k.size() * 2);
    dump(output / "V.f16", v.data(), v.size() * 2);
    std::vector<double> golden(S1 * D), scores(S2);
    for (int i = 0; i < S1; ++i) {
      double mx = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < S2; ++j) {
        double score = 0;
        for (int d = 0; d < D; ++d) score += double(qf[i * D + d]) * kf[j * D + d];
        scores[j] = score / std::sqrt(double(D));
        mx = std::max(mx, scores[j]);
      }
      double sum = 0;
      for (auto &score : scores) { score = std::exp(score - mx); sum += score; }
      for (int d = 0; d < D; ++d) {
        double value = 0;
        for (int j = 0; j < S2; ++j) value += scores[j] * vf[j * D + d];
        golden[i * D + d] = value / sum;
      }
    }
    check(aclInit(nullptr), "Initialize ACL");
    check(aclrtSetDevice(device), "Select device");
    check(aclrtSetOpExecuteTimeOut(10), "Set kernel timeout");
    aclrtStream stream = nullptr;
    check(aclrtCreateStream(&stream), "Create stream");
    std::printf("SOC=%s case=%s seed=%s block_dim=1 mix_block_dim=2\n",
                aclrtGetSocName(), mode.c_str(), argv[3]);
    std::vector<float> values(S1 * D, std::numeric_limits<float>::quiet_NaN());
    double maxError = 0, squaredError = 0;
    size_t bad = 0;
    {
      Buffer qd(q.data(), q.size() * 2), kd(k.data(), k.size() * 2);
      Buffer vd(v.data(), v.size() * 2), od(values.data(), values.size() * 4);
      KERNEL_SYMBOL(1, 0, stream, qd.payload(), kd.payload(),
                                   vd.payload(), od.payload(), S1, S2);
      check(aclrtSynchronizeStream(stream), "Wait for kernel");
      qd.read(true); kd.read(true); vd.read(true);
      auto actual = od.read(false);
      std::memcpy(values.data(), actual.data() + Buffer::guard, values.size() * 4);
    }
    dump(output / "O.f32", values.data(), values.size() * 4);
    for (size_t i = 0; i < values.size(); ++i) {
      const double error = std::isfinite(values[i]) ? std::fabs(values[i] - golden[i])
                         : std::numeric_limits<double>::infinity();
      const double tolerance = mode == "pattern" ? 1e-4 + 1e-3 * std::fabs(golden[i]) : 1e-6;
      maxError = std::max(maxError, error);
      squaredError += error * error;
      if (error > tolerance) {
        if (bad < 8) std::printf("Mismatch index=%zu actual=%.9g expected=%.12g\n", i, values[i], golden[i]);
        ++bad;
      }
    }
    std::printf("mismatches=%zu max_abs_error=%.12g rmse=%.12g input_unchanged=1 guard_unchanged=1\n",
                bad, maxError, std::sqrt(squaredError / values.size()));
    check(aclrtDestroyStream(stream), "Destroy stream");
    check(aclrtResetDevice(device), "Release device");
    check(aclFinalize(), "Finalize ACL");
    std::puts(bad ? "NUMERICAL_FAIL" : "NUMERICAL_PASS");
    return bad ? 3 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
