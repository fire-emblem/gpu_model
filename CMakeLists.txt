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

add_library(gpu_model STATIC
  src/arch/arch_registry.cpp
  src/arch/c500_spec.cpp
  src/decode/generated_gcn_full_opcode_table.cpp
  src/decode/generated_gcn_inst_db.cpp
  src/decode/generated_gcn_opcode_enums.cpp
  src/decode/gcn_inst_db_lookup.cpp
  src/decode/gcn_inst_format.cpp
  src/decode/gcn_inst_encoding_def.cpp
  src/decode/gcn_inst_decoder.cpp
  src/decode/gcn_inst_formatter.cpp
  src/isa/instruction_builder.cpp
  src/isa/kernel_metadata.cpp
  src/isa/kernel_program.cpp
  src/isa/opcode_descriptor.cpp
  src/isa/target_isa.cpp
  src/loader/asm_parser.cpp
  src/loader/amdgpu_code_object_decoder.cpp
  src/loader/amdgpu_obj_loader.cpp
  src/loader/device_image_loader.cpp
  src/loader/device_segment_image.cpp
  src/loader/executable_image_io.cpp
  src/loader/gcn_lowering_rules.cpp
  src/loader/gcn_text_parser.cpp
  src/loader/program_lowering.cpp
  src/loader/program_bundle_io.cpp
  src/loader/program_file_loader.cpp
  src/memory/memory_system.cpp
  src/memory/cache_model.cpp
  src/memory/shared_bank_model.cpp
  src/exec/opcode_execution_info.cpp
  src/exec/raw_gcn_executor.cpp
  src/exec/raw_gcn_semantic_handlers.cpp
  src/exec/semantic_handlers.cpp
  src/exec/semantics.cpp
  src/exec/issue_model.cpp
  src/exec/issue_scheduler.cpp
  src/exec/issue_eligibility.cpp
  src/exec/functional_executor.cpp
  src/exec/functional_execution_core.cpp
  src/exec/parallel_wave_executor.cpp
  src/exec/scoreboard.cpp
  src/exec/event_queue.cpp
  src/exec/handlers/compute_functor_registry.cpp
  src/exec/cycle_executor.cpp
  src/runtime/mapper.cpp
  src/runtime/host_runtime.cpp
  src/runtime/hip_interposer_state.cpp
  src/runtime/runtime_hooks.cpp
  src/debug/instruction_trace.cpp
  src/debug/cycle_timeline.cpp
  src/debug/debug_info_io.cpp
  src/debug/trace_sink.cpp
)

set_target_properties(gpu_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(gpu_model
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(GPU_MODEL_ENABLE_MARL "Enable marl-backed parallel execution scaffolding" ON)

if(GPU_MODEL_ENABLE_MARL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marl/CMakeLists.txt")
  set(MARL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MARL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MARL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MARL_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
  set(_gpu_model_saved_cxx_standard ${CMAKE_CXX_STANDARD})
  add_subdirectory(third_party/marl EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_STANDARD ${_gpu_model_saved_cxx_standard})
  target_link_libraries(gpu_model PUBLIC marl)
  target_compile_definitions(gpu_model PUBLIC GPU_MODEL_HAS_MARL=1)
endif()

gpu_model_enable_warnings(gpu_model)

add_subdirectory(tests)

add_executable(vecadd_main
  examples/vecadd_main.cpp
)

target_link_libraries(vecadd_main PRIVATE gpu_model)
gpu_model_enable_warnings(vecadd_main)

if(EXISTS "/opt/rocm/include/hip/hip_runtime_api.h")
  add_library(gpu_model_hip_interposer SHARED
    src/runtime/hip_interposer.cpp
  )
  target_include_directories(gpu_model_hip_interposer PRIVATE /opt/rocm/include)
  target_link_libraries(gpu_model_hip_interposer PRIVATE gpu_model)
  target_compile_definitions(gpu_model_hip_interposer PRIVATE _GNU_SOURCE __HIP_PLATFORM_AMD__)
  target_link_options(gpu_model_hip_interposer PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_interposer.version")
  gpu_model_enable_warnings(gpu_model_hip_interposer)
endif()

add_executable(fma_loop_cycle_trace_main
  examples/fma_loop_cycle_trace_main.cpp
)

target_link_libraries(fma_loop_cycle_trace_main PRIVATE gpu_model)
gpu_model_enable_warnings(fma_loop_cycle_trace_main)

add_executable(code_object_dump_main
  examples/code_object_dump_main.cpp
)

target_link_libraries(code_object_dump_main PRIVATE gpu_model)
gpu_model_enable_warnings(code_object_dump_main)
