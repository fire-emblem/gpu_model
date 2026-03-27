#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "gpu_model/decode/gcn_inst_formatter.h"
#include "gpu_model/loader/amdgpu_code_object_decoder.h"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " <amdgpu-object-or-hip-out> [kernel_name]\n";
    return 1;
  }

  const std::filesystem::path path = argv[1];
  std::optional<std::string> kernel_name;
  if (argc == 3) {
    kernel_name = argv[2];
  }

  try {
    const auto image =
        gpu_model::AmdgpuCodeObjectDecoder{}.Decode(path, kernel_name);
    gpu_model::GcnInstFormatter formatter;

    std::cout << "kernel=" << image.kernel_name << '\n';
    for (const auto& instruction : image.instructions) {
      std::cout << "pc=0x" << std::hex << instruction.pc << std::dec
                << " size=" << instruction.size_bytes
                << " fmt=" << gpu_model::ToString(instruction.format_class)
                << " text=" << formatter.Format(instruction) << '\n';
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "decode failed: " << ex.what() << '\n';
    return 2;
  }
}
