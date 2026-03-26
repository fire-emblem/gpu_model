add_executable(gpu_model_tests
  arch/arch_registry_test.cpp
  isa/kernel_program_test.cpp
  state/wave_state_test.cpp
  functional/mapper_test.cpp
  functional/vecadd_functional_test.cpp
  functional/predicated_if_functional_test.cpp
  functional/shared_barrier_functional_test.cpp
  functional/private_memory_functional_test.cpp
  cycle/cycle_smoke_test.cpp
  cycle/async_memory_cycle_test.cpp
  cycle/shared_barrier_cycle_test.cpp
  cycle/private_memory_cycle_test.cpp
  loader/asm_parser_test.cpp
  runtime/trace_test.cpp
)

target_link_libraries(gpu_model_tests
  PRIVATE
    gpu_model
    GTest::gtest_main
)

gpu_model_enable_warnings(gpu_model_tests)

include(GoogleTest)
gtest_discover_tests(gpu_model_tests)
