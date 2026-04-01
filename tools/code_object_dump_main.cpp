#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "gpu_model/instruction/encoded/encoded_instruction_formatter.h"
#include "gpu_model/program/object_reader.h"

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
        gpu_model::ObjectReader{}.LoadEncodedObject(path, kernel_name);
    gpu_model::EncodedInstructionFormatter formatter;

    std::cout << "kernel=" << image.kernel_name << '\n';
    for (size_t i = 0; i < image.decoded_instructions.size(); ++i) {
      const auto& instruction = image.decoded_instructions[i];
      std::cout << "pc=0x" << std::hex << instruction.pc << std::dec
                << " size=" << instruction.size_bytes
                << " fmt=" << gpu_model::ToString(instruction.format_class)
                << " op_type="
                << (i < image.instruction_objects.size() && image.instruction_objects[i] != nullptr
                        ? image.instruction_objects[i]->op_type_name()
                        : "none")
                << " class="
                << (i < image.instruction_objects.size() && image.instruction_objects[i] != nullptr
                        ? image.instruction_objects[i]->class_name()
                        : "none")
                << " text=" << formatter.Format(instruction) << '\n';
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "decode failed: " << ex.what() << '\n';
    return 2;
  }
}
