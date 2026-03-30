#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_model/program/object_reader.h"

namespace gpu_model {
namespace {

bool HasLlvmMcAmdgpuToolchain() {
  return std::system("command -v llvm-mc >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string ShellQuote(const std::filesystem::path& path) {
  return "'" + path.string() + "'";
}

struct AssembledModule {
  std::filesystem::path temp_dir;
  std::filesystem::path asm_path;
  std::filesystem::path obj_path;
  EncodedProgramObject image;
};

std::optional<std::string> ExtractFixtureDirective(const std::string& text, std::string_view key);

AssembledModule AssembleAndDecodeLlvmMcModule(const std::string& stem,
                                              const std::string& kernel_name,
                                              const std::string& assembly_text,
                                              const std::string& mcpu = "gfx900") {
  const auto temp_dir = MakeUniqueTempDir(stem);
  const auto asm_path = temp_dir / (kernel_name + ".s");
  const auto obj_path = temp_dir / (kernel_name + ".o");
  {
    std::ofstream out(asm_path);
    if (!out) {
      throw std::runtime_error("failed to create asm file: " + asm_path.string());
    }
    out << assembly_text;
  }

  const std::string assemble_command =
      "llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=" + mcpu + " -filetype=obj " +
      ShellQuote(asm_path) + " -o " + ShellQuote(obj_path);
  if (std::system(assemble_command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for asm module: " + kernel_name);
  }

  return AssembledModule{
      .temp_dir = temp_dir,
      .asm_path = asm_path,
      .obj_path = obj_path,
      .image = ObjectReader{}.LoadEncodedObject(obj_path, kernel_name),
  };
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

AssembledModule AssembleAndDecodeLlvmMcModuleFromFixture(const std::string& stem,
                                                         const std::string& kernel_name,
                                                         const std::filesystem::path& fixture_path) {
  const auto text = ReadTextFile(fixture_path);
  const auto mcpu = ExtractFixtureDirective(text, "GPU_MODEL_MCPU").value_or("gfx900");
  return AssembleAndDecodeLlvmMcModule(stem, kernel_name, text, mcpu);
}

std::optional<std::string> ExtractFixtureDirective(const std::string& text, std::string_view key) {
  const std::string needle = std::string(key) + ":";
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.starts_with("#")) {
      continue;
    }
    const auto pos = line.find(needle);
    if (pos == std::string::npos) {
      continue;
    }
    std::string value = line.substr(pos + needle.size());
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) {
      return std::string{};
    }
    const auto last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
  }
  return std::nullopt;
}

std::string KernelNameForFixture(const std::filesystem::path& fixture_path) {
  const std::string text = ReadTextFile(fixture_path);
  if (const auto explicit_name = ExtractFixtureDirective(text, "GPU_MODEL_KERNEL");
      explicit_name.has_value() && !explicit_name->empty()) {
    return *explicit_name;
  }

  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.starts_with(".globl ")) {
      return line.substr(std::string(".globl ").size());
    }
  }
  throw std::runtime_error("missing kernel name directive or .globl in fixture: " + fixture_path.string());
}

std::optional<std::string> SkipReasonForFixture(const std::filesystem::path& fixture_path) {
  return ExtractFixtureDirective(ReadTextFile(fixture_path), "GPU_MODEL_SKIP");
}

std::vector<std::string> ExpectedMnemonicsForFixture(const std::filesystem::path& fixture_path) {
  const auto raw = ExtractFixtureDirective(ReadTextFile(fixture_path), "GPU_MODEL_EXPECT_MNEMONICS");
  if (!raw.has_value() || raw->empty()) {
    return {};
  }
  std::vector<std::string> values;
  std::istringstream stream(*raw);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const auto first = item.find_first_not_of(" \t");
    if (first == std::string::npos) {
      continue;
    }
    const auto last = item.find_last_not_of(" \t");
    values.push_back(item.substr(first, last - first + 1));
  }
  return values;
}

std::vector<std::filesystem::path> LoaderAsmFixtures() {
  std::vector<std::filesystem::path> fixtures;
  for (const auto& entry : std::filesystem::directory_iterator("tests/asm_cases/loader")) {
    if (entry.is_regular_file() && entry.path().extension() == ".s") {
      fixtures.push_back(entry.path());
    }
  }
  std::sort(fixtures.begin(), fixtures.end());
  return fixtures;
}

std::string FixtureNameForGtest(const std::filesystem::path& path) {
  std::string name = path.stem().string();
  for (char& ch : name) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_')) {
      ch = '_';
    }
  }
  return name;
}

TEST(AsmModuleIntegrationTest, DecodesModuleFromLlvmMcAssembledAmdgpuAssembly) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled = AssembleAndDecodeLlvmMcModuleFromFixture(
      "gpu_model_llvm_mc_asm_module",
      "asm_module_probe",
      std::filesystem::path("tests/asm_cases/loader/asm_module_probe.s"));
  const auto& image = assembled.image;
  EXPECT_EQ(image.kernel_name, "asm_module_probe");
  ASSERT_FALSE(image.instructions.empty());
  EXPECT_EQ(image.metadata.values.at("entry"), "asm_module_probe");
  EXPECT_EQ(image.metadata.values.at("descriptor_symbol"), "asm_module_probe.kd");

  const auto mnemonics = [&]() {
    std::vector<std::string> values;
    values.reserve(image.instructions.size());
    for (const auto& inst : image.instructions) {
      values.push_back(inst.mnemonic);
    }
    return values;
  }();

  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "s_load_dwordx2"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "s_waitcnt"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "v_mov_b32_e32"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "v_add_f32_e32"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "s_add_u32"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "s_cmp_lt_i32"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "s_cbranch_scc0"), mnemonics.end());
  EXPECT_NE(std::find(mnemonics.begin(), mnemonics.end(), "global_store_dword"), mnemonics.end());
  EXPECT_EQ(mnemonics.back(), "s_endpgm");

  const auto cmp_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "s_cmp_lt_i32"; });
  ASSERT_NE(cmp_it, image.instructions.end());
  ASSERT_EQ(cmp_it->decoded_operands.size(), 2u);
  EXPECT_EQ(cmp_it->decoded_operands[0].text, "s4");
  EXPECT_EQ(cmp_it->decoded_operands[1].text, "16");

  const auto branch_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "s_cbranch_scc0"; });
  ASSERT_NE(branch_it, image.instructions.end());
  ASSERT_EQ(branch_it->decoded_operands.size(), 1u);
  EXPECT_TRUE(branch_it->decoded_operands[0].info.has_immediate);

  const auto store_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "global_store_dword"; });
  ASSERT_NE(store_it, image.instructions.end());
  ASSERT_EQ(store_it->decoded_operands.size(), 4u);
  EXPECT_EQ(store_it->decoded_operands[0].text, "v1");
  EXPECT_EQ(store_it->decoded_operands[1].text, "s[0:1]");
  EXPECT_EQ(store_it->decoded_operands[2].text, "v3");
  EXPECT_EQ(store_it->decoded_operands[3].text, "off");
}

TEST(AsmModuleIntegrationTest, DecodesVariantHeavyLlvmMcAssemblyModule) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled = AssembleAndDecodeLlvmMcModuleFromFixture(
      "gpu_model_llvm_mc_asm_variant_module",
      "asm_variant_probe",
      std::filesystem::path("tests/asm_cases/loader/asm_variant_probe.s"));
  const auto& image = assembled.image;

  EXPECT_EQ(image.kernel_name, "asm_variant_probe");
  EXPECT_EQ(image.metadata.values.at("entry"), "asm_variant_probe");
  EXPECT_EQ(image.metadata.values.at("descriptor_symbol"), "asm_variant_probe.kd");
  EXPECT_FALSE(image.instructions.empty());

  const auto expect_present = [&](std::string_view mnemonic) {
    EXPECT_NE(std::find_if(image.instructions.begin(), image.instructions.end(),
                           [&](const EncodedGcnInstruction& inst) { return inst.mnemonic == mnemonic; }),
              image.instructions.end());
  };

  expect_present("s_mov_b32");
  expect_present("s_movk_i32");
  expect_present("s_add_u32");
  expect_present("s_cmp_lt_i32");
  expect_present("s_cbranch_scc0");
  expect_present("v_mov_b32_e32");
  expect_present("v_add_f32_e32");
  expect_present("v_max_f32_e32");
  EXPECT_EQ(image.instructions.back().mnemonic, "s_endpgm");

  const auto movk_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "s_movk_i32"; });
  ASSERT_NE(movk_it, image.instructions.end());
  ASSERT_EQ(movk_it->decoded_operands.size(), 2u);
  EXPECT_EQ(movk_it->decoded_operands[0].text, "s5");
  EXPECT_EQ(movk_it->decoded_operands[1].text, "-9");

  const auto cmp_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "s_cmp_lt_i32"; });
  ASSERT_NE(cmp_it, image.instructions.end());
  ASSERT_EQ(cmp_it->decoded_operands.size(), 2u);
  EXPECT_EQ(cmp_it->decoded_operands[0].text, "s6");
  EXPECT_EQ(cmp_it->decoded_operands[1].text, "1");

  const auto vmax_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "v_max_f32_e32"; });
  ASSERT_NE(vmax_it, image.instructions.end());
  ASSERT_EQ(vmax_it->decoded_operands.size(), 3u);
  EXPECT_EQ(vmax_it->decoded_operands[0].text, "v4");
  EXPECT_EQ(vmax_it->decoded_operands[1].text, "v1");
  EXPECT_EQ(vmax_it->decoded_operands[2].text, "v2");
}

TEST(AsmModuleIntegrationTest, DecodesFlatAndAtomicLlvmMcAssemblyModule) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled = AssembleAndDecodeLlvmMcModuleFromFixture(
      "gpu_model_llvm_mc_asm_flat_module",
      "asm_flat_variants",
      std::filesystem::path("tests/asm_cases/loader/asm_flat_variants.s"));
  const auto& image = assembled.image;

  EXPECT_EQ(image.kernel_name, "asm_flat_variants");
  EXPECT_EQ(image.metadata.values.at("entry"), "asm_flat_variants");
  EXPECT_EQ(image.metadata.values.at("descriptor_symbol"), "asm_flat_variants.kd");
  EXPECT_FALSE(image.instructions.empty());

  const auto flat_load_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "global_load_dword"; });
  ASSERT_NE(flat_load_it, image.instructions.end());
  ASSERT_EQ(flat_load_it->decoded_operands.size(), 4u);
  EXPECT_EQ(flat_load_it->decoded_operands[0].text, "v4");
  EXPECT_EQ(flat_load_it->decoded_operands[1].text, "v1");
  EXPECT_EQ(flat_load_it->decoded_operands[2].text, "s[0:1]");
  EXPECT_EQ(flat_load_it->decoded_operands[3].text, "off");

  const auto flat_store_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "global_store_dword"; });
  ASSERT_NE(flat_store_it, image.instructions.end());
  ASSERT_EQ(flat_store_it->decoded_operands.size(), 4u);
  EXPECT_EQ(flat_store_it->decoded_operands[0].text, "v1");
  EXPECT_EQ(flat_store_it->decoded_operands[1].text, "s[0:1]");
  EXPECT_EQ(flat_store_it->decoded_operands[2].text, "v4");
  EXPECT_EQ(flat_store_it->decoded_operands[3].text, "off");

  const auto atomic_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const EncodedGcnInstruction& inst) { return inst.mnemonic == "global_atomic_add"; });
  ASSERT_NE(atomic_it, image.instructions.end());
  ASSERT_EQ(atomic_it->decoded_operands.size(), 3u);
  EXPECT_EQ(atomic_it->decoded_operands[0].text, "v5");
  EXPECT_EQ(atomic_it->decoded_operands[1].text, "v6");
  EXPECT_EQ(atomic_it->decoded_operands[2].text, "s[0:1]");
}

class LoaderAsmFixtureTest : public ::testing::TestWithParam<std::filesystem::path> {};

TEST_P(LoaderAsmFixtureTest, AssemblesAndDecodesFixtureModule) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto& fixture_path = GetParam();
  if (const auto skip_reason = SkipReasonForFixture(fixture_path); skip_reason.has_value()) {
    GTEST_SKIP() << *skip_reason;
  }

  const auto kernel_name = KernelNameForFixture(fixture_path);
  const auto assembled = AssembleAndDecodeLlvmMcModuleFromFixture(
      "gpu_model_loader_asm_fixture", kernel_name, fixture_path);
  const auto& image = assembled.image;

  EXPECT_EQ(image.kernel_name, kernel_name);
  EXPECT_FALSE(image.instructions.empty());
  EXPECT_EQ(image.metadata.values.at("entry"), kernel_name);
  EXPECT_EQ(image.instructions.back().mnemonic, "s_endpgm");

  const auto expected_mnemonics = ExpectedMnemonicsForFixture(fixture_path);
  for (const auto& mnemonic : expected_mnemonics) {
    EXPECT_NE(std::find_if(image.instructions.begin(), image.instructions.end(),
                           [&](const EncodedGcnInstruction& inst) { return inst.mnemonic == mnemonic; }),
              image.instructions.end())
        << "missing mnemonic " << mnemonic << " in fixture " << fixture_path;
  }
}

INSTANTIATE_TEST_SUITE_P(
    LoaderAsmFixtures,
    LoaderAsmFixtureTest,
    ::testing::ValuesIn(LoaderAsmFixtures()),
    [](const ::testing::TestParamInfo<std::filesystem::path>& info) {
      return FixtureNameForGtest(info.param);
    });

}  // namespace
}  // namespace gpu_model
