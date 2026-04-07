cmake_minimum_required(VERSION 3.22)

project(gpu_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CONFIGURATION_TYPES)
  set(_gpu_model_enable_asan_default OFF)
else()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_gpu_model_enable_asan_default ON)
  else()
    set(_gpu_model_enable_asan_default OFF)
  endif()
endif()
option(GPU_MODEL_ENABLE_ASAN "Enable AddressSanitizer" ${_gpu_model_enable_asan_default})

find_package(Threads REQUIRED)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "" FORCE)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

find_program(MOLD_PROGRAM mold)
if(MOLD_PROGRAM AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_link_options("-fuse-ld=mold")
  message(STATUS "Using mold linker: ${MOLD_PROGRAM}")
endif()

if(GPU_MODEL_ENABLE_ASAN)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "GPU_MODEL_ENABLE_ASAN requires GCC or Clang")
  endif()
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address -fno-omit-frame-pointer)
  message(STATUS "AddressSanitizer enabled")
endif()

include(cmake/warnings.cmake)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

add_library(gpu_model_loguru STATIC
  third_party/loguru/loguru.cpp
)
set_target_properties(gpu_model_loguru PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gpu_model_loguru
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/loguru
)
target_link_libraries(gpu_model_loguru
  PUBLIC
    Threads::Threads
    dl
)

add_library(gpu_model STATIC
  src/arch/arch_registry.cpp
  src/arch/c500_spec.cpp
  src/instruction/encoded/internal/generated_encoded_gcn_full_opcode_table.cpp
  src/instruction/encoded/internal/generated_encoded_gcn_inst_db.cpp
  src/instruction/encoded/internal/encoded_gcn_db_lookup.cpp
  src/instruction/encoded/encoded_gcn_inst_format.cpp
  src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp
  src/instruction/encoded/instruction_decoder.cpp
  src/instruction/encoded/encoded_instruction_formatter.cpp
  src/isa/instruction_builder.cpp
  src/isa/kernel_metadata.cpp
  src/isa/kernel_program_builder.cpp
  src/isa/opcode_descriptor.cpp
  src/loader/asm_parser.cpp
  src/loader/device_image_loader.cpp
  src/loader/device_segment_image.cpp
  src/loader/executable_image_io.cpp
  src/loader/gcn_text_parser.cpp
  src/loader/program_bundle_io.cpp
  src/program/encoded_program_object.cpp
  src/program/object_reader.cpp
  src/memory/memory_system.cpp
  src/memory/cache_model.cpp
  src/memory/shared_bank_model.cpp
  src/execution/internal/opcode_execution_info.cpp
  src/instruction/encoded/internal/encoded_instruction_descriptor.cpp
  src/instruction/encoded/internal/encoded_instruction_binding.cpp
  src/instruction/encoded/instruction_object.cpp
  src/execution/program_object_exec_engine.cpp
  src/execution/encoded_semantic_handler.cpp
  src/execution/memory_ops.cpp
  src/execution/sync_ops.cpp
  src/execution/plan_apply.cpp
  src/execution/internal/semantic_handler.cpp
  src/execution/internal/semantics.cpp
  src/execution/wave_context_builder.cpp
  src/execution/internal/issue_model.cpp
  src/execution/internal/issue_scheduler.cpp
  src/execution/internal/issue_eligibility.cpp
  src/execution/internal/async_scoreboard.cpp
  src/execution/functional_exec_engine.cpp
  src/execution/internal/event_queue.cpp
  src/execution/internal/handlers/compute_functor_registry.cpp
  src/execution/cycle_exec_engine.cpp
  src/runtime/mapper.cpp
  src/runtime/kernarg_packer.cpp
  src/runtime/program_cycle_tracker.cpp
  src/runtime/exec_engine.cpp
  src/runtime/hip_runtime.cpp
  src/runtime/core/module_registry.cpp
  src/runtime/core/model_runtime.cpp
  src/runtime/core/device_memory_manager.cpp
  src/runtime/core/runtime_session.cpp
  src/runtime/config/runtime_env_config.cpp
  src/runtime/logging/runtime_log_service.cpp
  src/debug/trace/instruction_trace.cpp
  src/debug/recorder/recorder.cpp
  src/debug/recorder/recorder_export.cpp
  src/debug/recorder/trace_recorder_adapter.cpp
  src/debug/trace/trace_artifact_recorder.cpp
  src/debug/trace/trace_format.cpp
  src/debug/trace/trace_json_fields.cpp
  src/debug/timeline/trace_perfetto_proto.cpp
  src/debug/trace/trace_event_export.cpp
  src/debug/trace/trace_event_view.cpp
  src/debug/timeline/cycle_timeline_layout.cpp
  src/debug/timeline/cycle_timeline_google_trace.cpp
  src/debug/timeline/cycle_timeline_perfetto.cpp
  src/debug/timeline/cycle_timeline.cpp
  src/debug/info/debug_info_io.cpp
  src/debug/trace/trace_sink.cpp
)

set_target_properties(gpu_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(gpu_model
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gpu_model
  PUBLIC
    gpu_model_loguru
)

gpu_model_enable_warnings(gpu_model)

add_executable(gpu_model_perfetto_waitcnt_slots_demo
  examples/11-perfetto-waitcnt-slots/perfetto_waitcnt_slots_demo.cpp
)
target_link_libraries(gpu_model_perfetto_waitcnt_slots_demo PRIVATE gpu_model)
target_include_directories(gpu_model_perfetto_waitcnt_slots_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
gpu_model_enable_warnings(gpu_model_perfetto_waitcnt_slots_demo)

add_subdirectory(tests)

if(EXISTS "/opt/rocm/include/hip/hip_runtime_api.h")
  add_library(gpu_model_hip_runtime_abi SHARED
    src/runtime/hip_runtime_abi.cpp
  )
  target_include_directories(gpu_model_hip_runtime_abi PRIVATE /opt/rocm/include)
  target_link_libraries(gpu_model_hip_runtime_abi PRIVATE gpu_model)
  target_compile_definitions(gpu_model_hip_runtime_abi PRIVATE _GNU_SOURCE __HIP_PLATFORM_AMD__)
  target_link_options(gpu_model_hip_runtime_abi PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_runtime_abi.version")
  gpu_model_enable_warnings(gpu_model_hip_runtime_abi)
  if(TARGET gpu_model_tests)
    add_dependencies(gpu_model_tests gpu_model_hip_runtime_abi)
  endif()
endif()
