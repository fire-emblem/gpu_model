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
  URL_HASH SHA256=f179ec217f9b3b3f3c6e8b02d3e7eda997b49e4ce26d6b235c9053bec9c0bf9f
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
  # gpu_arch/chip_config/
  src/gpu_arch/chip_config/arch_registry.cpp
  src/gpu_arch/chip_config/mac500_spec.cpp
  # gpu_arch/issue_config/
  src/gpu_arch/issue_config/issue_config.cpp
  # instruction/isa/
  src/instruction/isa/instruction.cpp
  src/instruction/isa/instruction_builder.cpp
  src/instruction/isa/kernel_metadata.cpp
  src/instruction/isa/kernel_program_builder.cpp
  src/instruction/isa/opcode_descriptor.cpp
  src/instruction/isa/opcode_info.cpp
  # instruction/decode/encoded/
  src/instruction/decode/encoded/internal/generated_encoded_gcn_full_opcode_table.cpp
  src/instruction/decode/encoded/internal/generated_encoded_gcn_inst_db.cpp
  src/instruction/decode/encoded/internal/encoded_gcn_db_lookup.cpp
  src/instruction/decode/encoded/encoded_gcn_inst_format.cpp
  src/instruction/decode/encoded/internal/encoded_gcn_encoding_def.cpp
  src/instruction/decode/encoded/decoded_instruction.cpp
  src/instruction/decode/encoded/instruction_decoder.cpp
  src/instruction/decode/encoded/encoded_instruction_formatter.cpp
  src/instruction/decode/encoded/internal/encoded_instruction_descriptor.cpp
  src/instruction/decode/encoded/internal/encoded_instruction_binding.cpp
  src/instruction/decode/encoded/instruction_object.cpp
  # instruction/semantics/
  src/instruction/semantics/encoded_handler.cpp
  src/instruction/semantics/branch/branch_handlers.cpp
  src/instruction/semantics/memory/memory_handlers.cpp
  src/instruction/semantics/scalar/scalar_handlers.cpp
  src/instruction/semantics/vector/vector_handlers.cpp
  # program/loader/
  src/program/loader/asm_parser.cpp
  src/program/loader/device_image_loader.cpp
  src/program/loader/device_segment_image.cpp
  src/program/loader/executable_image_io.cpp
  src/program/loader/external_tool_executor.cpp
  src/program/loader/gcn_text_parser.cpp
  src/program/loader/amdgpu_target_config.cpp
  src/program/loader/artifact_parser.cpp
  src/program/loader/code_object_binding.cpp
  src/program/loader/code_object_builder.cpp
  src/program/loader/code_object_disassembly.cpp
  src/program/loader/code_object_materializer.cpp
  src/program/loader/elf_section_loader.cpp
  src/program/loader/temp_dir_manager.cpp
  src/program/loader/program_bundle_io.cpp
  # program/encoded/
  src/program/encoded/encoded_program_object.cpp
  # program/program_object/
  src/program/program_object/object_reader.cpp
  # execution/functional/
  src/execution/functional/functional_exec_engine.cpp
  # execution/cycle/
  src/execution/cycle/cycle_exec_engine.cpp
  # execution/encoded/
  src/execution/encoded/program_object_exec_engine.cpp
  # execution/internal/
  src/execution/internal/commit_logic/memory_ops.cpp
  src/execution/internal/sync_ops/sync_ops.cpp
  src/execution/internal/commit_logic/plan_apply.cpp
  src/execution/internal/plan/semantic_handler.cpp
  src/execution/internal/plan/semantics.cpp
  src/execution/internal/block_schedule/wave_context_builder.cpp
  src/execution/internal/issue_logic/issue_scheduler.cpp
  src/execution/internal/issue_logic/issue_eligibility.cpp
  src/execution/internal/commit_logic/async_scoreboard.cpp
  src/execution/internal/cost_model/cycle_types.cpp
  src/execution/internal/wave_schedule/cycle_wave_schedule.cpp
  src/execution/internal/issue_logic/cycle_issue_schedule.cpp
  src/execution/internal/wave_schedule/event_queue.cpp
  src/execution/internal/cost_model/compute_functor_registry.cpp
  # runtime/hip_runtime/
  src/runtime/hip_runtime/hip_runtime.cpp
  # runtime/model_runtime/
  src/runtime/model_runtime/model_runtime.cpp
  src/runtime/model_runtime/module_registry.cpp
  src/runtime/model_runtime/device_memory_manager.cpp
  src/runtime/model_runtime/runtime_abi_arg_packer.cpp
  src/runtime/model_runtime/runtime_abi_allocation_ops.cpp
  src/runtime/model_runtime/runtime_abi_memory_ops.cpp
  src/runtime/model_runtime/runtime_last_error_state.cpp
  src/runtime/model_runtime/runtime_kernel_symbol_state.cpp
  src/runtime/model_runtime/runtime_launch_config_state.cpp
  src/runtime/model_runtime/runtime_executable_launch_helper.cpp
  src/runtime/model_runtime/runtime_process_path.cpp
  src/runtime/model_runtime/runtime_submission_sync.cpp
  src/runtime/model_runtime/runtime_stream_event_state.cpp
  src/runtime/model_runtime/runtime_trace_state.cpp
  src/runtime/model_runtime/runtime_session.cpp
  # runtime/exec_engine/
  src/runtime/exec_engine/cycle_timing_config_resolver.cpp
  src/runtime/exec_engine/launch_request_validator.cpp
  src/runtime/exec_engine/launch_trace_emitter.cpp
  src/runtime/exec_engine/launch_dispatcher.cpp
  src/runtime/exec_engine/exec_engine.cpp
  # runtime/config/
  src/runtime/config/runtime_config.cpp
  src/runtime/config/runtime_env_config.cpp
  # runtime/config/
  src/runtime/config/kernarg_packer.cpp
  # runtime/model_runtime/
  src/runtime/model_runtime/mapper.cpp
  src/runtime/model_runtime/program_cycle_tracker.cpp
  # utils/logging/
  src/utils/logging/runtime_log_service.cpp
  # utils/config/
  src/utils/config/invocation.cpp
  # state/memory/
  src/state/wave/wave_stats.cpp
  src/state/memory/cache_model.cpp
  src/state/memory/memory_system.cpp
  src/state/memory/shared_bank_model.cpp
  # debug/trace/
  src/debug/trace/instruction_trace.cpp
  src/debug/trace/trace_artifact_recorder.cpp
  src/debug/trace/trace_format.cpp
  src/debug/trace/trace_json_fields.cpp
  src/debug/trace/trace_event_export.cpp
  src/debug/trace/trace_event_view.cpp
  src/debug/trace/trace_sink.cpp
  # debug/recorder/
  src/debug/recorder/recorder.cpp
  src/debug/recorder/recorder_export.cpp
  src/debug/recorder/trace_recorder_adapter.cpp
  # debug/timeline/
  src/debug/timeline/trace_perfetto_proto.cpp
  src/debug/timeline/cycle_timeline_layout.cpp
  src/debug/timeline/cycle_timeline_google_trace.cpp
  src/debug/timeline/cycle_timeline_perfetto.cpp
  src/debug/timeline/cycle_timeline.cpp
  src/debug/timeline/actual_timeline_builder.cpp
  src/debug/timeline/timeline_comparator.cpp
  # debug/info/
  src/debug/info/debug_info_io.cpp
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
  add_library(gpu_model_hip_ld_preload SHARED
    src/runtime/hip_runtime/hip_ld_preload.cpp
  )
  target_include_directories(gpu_model_hip_ld_preload PRIVATE /opt/rocm/include)
  target_link_libraries(gpu_model_hip_ld_preload PRIVATE gpu_model)
  target_compile_definitions(gpu_model_hip_ld_preload PRIVATE _GNU_SOURCE __HIP_PLATFORM_AMD__)
  target_link_options(gpu_model_hip_ld_preload PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_ld_preload.version")
  gpu_model_enable_warnings(gpu_model_hip_ld_preload)
  if(TARGET gpu_model_tests)
    add_dependencies(gpu_model_tests gpu_model_hip_ld_preload)
  endif()
endif()
