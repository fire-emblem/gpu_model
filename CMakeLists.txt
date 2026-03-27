cmake_minimum_required(VERSION 3.22)

project(gpu_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/warnings.cmake)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(gpu_model STATIC
  src/arch/arch_registry.cpp
  src/arch/c500_spec.cpp
  src/isa/instruction_builder.cpp
  src/isa/kernel_metadata.cpp
  src/isa/kernel_program.cpp
  src/loader/asm_parser.cpp
  src/loader/executable_image_io.cpp
  src/loader/program_bundle_io.cpp
  src/loader/program_file_loader.cpp
  src/memory/memory_system.cpp
  src/memory/cache_model.cpp
  src/memory/shared_bank_model.cpp
  src/exec/semantics.cpp
  src/exec/functional_executor.cpp
  src/exec/scoreboard.cpp
  src/exec/event_queue.cpp
  src/exec/cycle_executor.cpp
  src/runtime/mapper.cpp
  src/runtime/host_runtime.cpp
  src/runtime/runtime_hooks.cpp
  src/debug/instruction_trace.cpp
  src/debug/cycle_timeline.cpp
  src/debug/debug_info_io.cpp
  src/debug/trace_sink.cpp
)

target_include_directories(gpu_model
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

gpu_model_enable_warnings(gpu_model)

add_subdirectory(tests)

add_executable(vecadd_main
  examples/vecadd_main.cpp
)

target_link_libraries(vecadd_main PRIVATE gpu_model)
gpu_model_enable_warnings(vecadd_main)

add_executable(fma_loop_cycle_trace_main
  examples/fma_loop_cycle_trace_main.cpp
)

target_link_libraries(fma_loop_cycle_trace_main PRIVATE gpu_model)
gpu_model_enable_warnings(fma_loop_cycle_trace_main)
