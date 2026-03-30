#include <gtest/gtest.h>

#include <vector>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/exec/encoded/semantics/raw_gcn_semantic_handler.h"
#include "gpu_model/execution/wave_context.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"

namespace gpu_model {
namespace {

class InstructionObjectExecuteTest : public ::testing::Test {
 protected:
  struct Harness {
    WaveContext wave;
    uint64_t vcc = 0;
    std::vector<std::byte> kernarg;
    MemorySystem memory;
    ExecutionStats stats;
    std::vector<std::byte> shared_memory;
    uint64_t barrier_generation = 0;
    uint32_t barrier_arrivals = 0;
    RawGcnBlockContext block;
    RawGcnWaveContext context;

    Harness()
        : block{
              .shared_memory = shared_memory,
              .barrier_generation = barrier_generation,
              .barrier_arrivals = barrier_arrivals,
              .wave_count = 1,
          },
          context(wave, vcc, kernarg, 0, memory, stats, block) {
      wave.thread_count = kWaveSize;
      wave.ResetInitialExec();
    }
  };

  static InstructionObjectPtr ParseSingleObject(uint64_t pc,
                                                std::vector<uint32_t> words,
                                                GcnInstFormatClass format_class,
                                                std::string mnemonic) {
    RawGcnInstruction raw;
    raw.pc = pc;
    raw.words = std::move(words);
    raw.format_class = format_class;
    raw.size_bytes = raw.words.size() * sizeof(uint32_t);
    raw.mnemonic = std::move(mnemonic);
    DecodeGcnOperands(raw);
    auto parsed = InstructionArrayParser::Parse(std::vector<RawGcnInstruction>{raw});
    EXPECT_EQ(parsed.instruction_objects.size(), 1u);
    return std::move(parsed.instruction_objects.front());
  }
};

TEST_F(InstructionObjectExecuteTest, ExecutesWaitcntDirectly) {
  auto object = ParseSingleObject(0x1910, {0xbf8cc07fu}, GcnInstFormatClass::Sopp, "s_waitcnt");
  ASSERT_NE(object, nullptr);
  ASSERT_EQ(object->class_name(), "s_waitcnt");

  Harness harness;
  harness.wave.pc = 0x1910;
  object->Execute(harness.context);

  EXPECT_EQ(harness.wave.pc, 0x1914u);
  EXPECT_EQ(harness.wave.status, WaveStatus::Active);
}

TEST_F(InstructionObjectExecuteTest, ExecutesEndpgmDirectly) {
  auto object = ParseSingleObject(0x1994, {0xbf810000u}, GcnInstFormatClass::Sopp, "s_endpgm");
  ASSERT_NE(object, nullptr);
  ASSERT_EQ(object->class_name(), "s_endpgm");

  Harness harness;
  harness.wave.pc = 0x1994;
  object->Execute(harness.context);

  EXPECT_EQ(harness.wave.status, WaveStatus::Exited);
  EXPECT_EQ(harness.stats.wave_exits, 1u);
}

TEST_F(InstructionObjectExecuteTest, ExecutesExeczBranchDirectly) {
  auto object =
      ParseSingleObject(0x192c, {0xbf880019u}, GcnInstFormatClass::Sopp, "s_cbranch_execz");
  ASSERT_NE(object, nullptr);
  ASSERT_EQ(object->class_name(), "s_cbranch_execz");

  Harness taken;
  taken.wave.pc = 0x192c;
  taken.wave.exec.reset();
  object->Execute(taken.context);
  EXPECT_EQ(taken.wave.pc, 0x1994u);

  Harness not_taken;
  not_taken.wave.pc = 0x192c;
  not_taken.wave.exec.set(0);
  object->Execute(not_taken.context);
  EXPECT_EQ(not_taken.wave.pc, 0x1930u);
}

}  // namespace
}  // namespace gpu_model
