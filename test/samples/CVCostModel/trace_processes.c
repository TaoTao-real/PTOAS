/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * Linux-only acceptance instrumentation. Load into the orchestrator and all
 * dynamic child processes; record process starts before their main function.
 * This is diagnostic evidence, not an adversarial process sandbox.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

__attribute__((constructor)) static void record_start(void) {
  const char *path = getenv("PTOAS_CV_PROCESS_TRACE");
  if (!path) return;
  char exe[4096], args[8192], hex[16385], line[22000];
  ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n <= 0) _exit(121);
  exe[n] = 0;
  int input = open("/proc/self/cmdline", O_RDONLY);
  if (input < 0) _exit(122);
  ssize_t count = read(input, args, sizeof(args));
  close(input);
  if (count < 0 || count == (ssize_t)sizeof(args)) _exit(123);
  const char *digits = "0123456789abcdef";
  for (ssize_t i = 0; i < count; ++i) {
    hex[2*i] = digits[((unsigned char)args[i]) >> 4];
    hex[2*i+1] = digits[((unsigned char)args[i]) & 15];
  }
  hex[2*count] = 0;
  const char *base = strrchr(exe, '/'); base = base ? base + 1 : exe;
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts)) _exit(124);
  int length = snprintf(line, sizeof(line),
      "{\"event\":\"process_start\",\"monotonic_ns\":%lld,\"pid\":%d,\"ppid\":%d,\"executable\":\"%s\",\"argv_hex\":\"%s\"}\n",
      (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec, (int)getpid(), (int)getppid(), base, hex);
  if (length < 0 || length >= (int)sizeof(line)) _exit(125);
  int output = open(path, O_WRONLY | O_APPEND | O_CREAT, 0600);
  if (output < 0 || write(output, line, length) != length) _exit(126);
  close(output);
}
