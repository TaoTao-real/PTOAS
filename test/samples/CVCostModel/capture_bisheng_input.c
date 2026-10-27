// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Read-only Linux capture of the exact LLVM stdin and argv used for an object.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
__attribute__((constructor)) static void capture(void) {
  const char *dir=getenv("PTO_GAP_CAPTURE"); if(!dir) return;
  char cmd[32768]; int fd=open("/proc/self/cmdline",O_RDONLY); if(fd<0)return;
  ssize_t n=read(fd,cmd,sizeof(cmd)-1);close(fd);if(n<=0)return;cmd[n]=0;
  const char *base=strrchr(cmd,'/');base=base?base+1:cmd;
  if(strcmp(base,"bisheng"))return;
  int c=0,ir=0,input=0;
  for(char *s=cmd;s<cmd+n;s+=strlen(s)+1){if(!strcmp(s,"-c"))c=1;if(!strcmp(s,"ir"))ir=1;if(!strcmp(s,"-"))input=1;}
  if(!c||!ir||!input)return;
  char path[4096];snprintf(path,sizeof(path),"%s/%ld.argv0",dir,(long)getpid());
  fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)_exit(120);if(write(fd,cmd,n)!=n)_exit(121);close(fd);
  snprintf(path,sizeof(path),"%s/%ld.ll",dir,(long)getpid());
  fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)_exit(122);
  char buffer[65536];off_t at=0;ssize_t r;
  while((r=pread(STDIN_FILENO,buffer,sizeof(buffer),at))>0){if(write(fd,buffer,r)!=r)_exit(123);at+=r;}
  if(r<0)_exit(124);close(fd);
}
