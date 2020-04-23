#include <fstream>
#include <string>
#include <vector>

#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/serialization/import.h"
#include "torch/script.h"

void test_to_from_vulkan() {
  auto tcpu = torch::tensor(
      {{

          {{1, 2, 3, 4}, {5, 6, 7, 8}},

          {{-1, -2, -3, -4}, {-5, -6, -7, -8}}

      }},
      torch::kFloat);
  std::cout << "tcpu:" << tcpu << std::endl;
  std::cout << "tcpu.options():" << tcpu.options() << std::endl;

  auto tcpus = tcpu.as_strided({1, 2, 1, 2}, {1, 1, 2, 2}, 0);
  std::cout << "tcpus:" << tcpus << std::endl;
  std::cout << "tcpus.options():" << tcpus.options() << std::endl;

  std::cout << "===> tcpu.to_vulkan()" << std::endl;
  auto tv = tcpu.to_vulkan();

  std::cout << "tv.device():" << tv.device() << std::endl;
  std::cout << "tv.layout():" << tv.layout() << std::endl;
  std::cout << "tv.options():" << tv.options() << std::endl;

  std::cout << "===> tv.as_strided()" << std::endl;
  auto tvs = tv.as_strided({1, 2, 1, 2}, {1, 1, 2, 2}, 0);
  std::cout << "tvs:" << tvs << std::endl;
  std::cout << "tvs.options():" << tvs.options() << std::endl;

  std::cout << "===> tv.to_densea()" << std::endl;
  auto tcpuout = tv.to_dense();
  std::cout << "tcpuout:" << tcpuout << std::endl;
  std::cout << "tcpuout.options():" << tcpuout.options() << std::endl;
}

void test_upsample_nearest2d() {
  auto tcpu = torch::tensor(
      {{

          {{1, 2, 3}, {4, 5, 6}},

          {{-1, -2, -3}, {-4, -5, -6}}

      }},
      torch::kFloat);
  auto tv = tcpu.to_vulkan();
  auto tvout = at::upsample_nearest2d(tv, {4, 6});
  auto tcpuout = tvout.to_dense();
  std::cout << "tcpuout:" << tcpuout << std::endl;
}

void test_add() {
  auto tin0_cpu = torch::tensor(
      {{

          {{1, 2, 3}, {4, 5, 6}},

          {{-1, -2, -3}, {-4, -5, -6}}

      }},
      torch::kFloat);
  auto tin1_cpu = torch::tensor(
      {{

          {{10, 20, 30}, {40, 50, 60}},

          {{-10, -20, -30}, {-40, -50, -60}}

      }},
      torch::kFloat);

  auto tout_cpu_expected = at::add(tin0_cpu, tin1_cpu, 2.f);
  std::cout << "tout_cpu_expected:" << tout_cpu_expected << std::endl;
  auto tin0_v = tin0_cpu.to_vulkan();
  auto tin1_v = tin1_cpu.to_vulkan();
  auto tout_v = at::add(tin0_v, tin1_v, 2.f);
  auto tout_cpu = tout_v.to_dense();
  std::cout << "tout_cpu:" << tout_cpu << std::endl;
}

void test_conv() {
  auto tin_cpu = torch::tensor( // 1, 3, 3, 3
      {{
          // c_0
          {
              {1, 2, 3},
              {4, 5, 6},
              {7, 8, 9},
          },
          // c_1
          {
              {101, 102, 103},
              {104, 105, 106},
              {107, 108, 109},
          },
          // c_2
          {
              {1001, 1002, 1003},
              {1004, 1005, 1006},
              {1007, 1008, 1009},
          },
      }},
      torch::kFloat);

  auto tw_cpu = torch::tensor(
      {
          // 2, 3, 2, 2
          // oc_0 (f_0)
          {{
               // oc_0 c_0
               {1, 0},
               {0, 0},
           },
           {
               // oc_0 c_1
               {0, 1},
               {0, 0},
           },
           {
               // oc_0 c_2
               {0, 0},
               {1, 0},
           }},
          // oc_1 (f_1)
          {{
               // oc_1 c_0
               {-1, 0},
               {0, 0},
           },
           {
               // oc_1 c_1
               {0, -1},
               {0, 0},
           },
           {
               // oc_1 c_2
               {0, 0},
               {-1, 0},
           }},
      },
      torch::kFloat);
  auto tb_cpu = torch::tensor({0, 0}, torch::kFloat);
  int64_t groups = 1;

  auto tout_cpu_expected = at::conv2d(
      tin_cpu,
      tw_cpu,
      tb_cpu,
      c10::IntArrayRef{1}, // stride
      c10::IntArrayRef{0}, // padding
      c10::IntArrayRef{1}, // dilation
      groups);
  std::cout << "tout_cpu_expected:" << tout_cpu_expected << std::endl;
  auto tin_v = tin_cpu.to_vulkan();
  auto tout_v = at::conv2d(
      tin_v,
      tw_cpu,
      tb_cpu,
      {1}, // stride
      {0}, // padding
      {1}, // dilation,
      groups);
  auto tout_cpu = tout_v.to_dense();
  std::cout << "tout_cpu:" << tout_cpu << std::endl;
}

int main(int argc, char** argv) {
  test_upsample_nearest2d();
  test_add();
  test_conv();
  test_to_from_vulkan();
  return 0;
}
