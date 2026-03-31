add_executable(gpu_model_tests
  instruction/internal/generated_encoded_gcn_full_opcode_table_test.cpp
  instruction/internal/generated_encoded_gcn_inst_db_test.cpp
  instruction/internal/encoded_gcn_db_lookup_test.cpp
  arch/arch_registry_test.cpp
  instruction/instruction_decoder_test.cpp
  instruction/instruction_formatter_test.cpp
  instruction/encoded_gcn_inst_format_test.cpp

  # execution/*
  execution/execution_naming_test.cpp
  execution/memory_ops_test.cpp
  execution/wave_context_builder_test.cpp
  execution/sync_ops_test.cpp
  execution/plan_apply_test.cpp

  # instruction/execution migrated coverage
  execution/internal/tensor_op_utils_test.cpp
  execution/internal/issue_model_test.cpp
  execution/internal/issue_eligibility_test.cpp
  execution/internal/functional_wait_reason_test.cpp
  execution/internal/issue_scheduler_test.cpp
  instruction/instruction_array_parser_test.cpp
  instruction/encoded_instruction_binding_test.cpp
  instruction/encoded_instruction_descriptor_test.cpp
  instruction/instruction_object_execute_test.cpp
  execution/encoded_semantic_execute_test.cpp
  execution/encoded_semantic_handler_registry_test.cpp
  execution/internal/handlers/semantic_handler_test.cpp
  isa/executable_kernel_builder_test.cpp
  isa/executable_kernel_test.cpp
  isa/opcode_descriptor_test.cpp
  isa/target_isa_test.cpp
  execution/wave_context_test.cpp
  functional/mapper_test.cpp
  functional/vecadd_functional_test.cpp
  functional/predicated_if_functional_test.cpp
  functional/shared_barrier_functional_test.cpp
  functional/private_memory_functional_test.cpp
  functional/constant_memory_functional_test.cpp
  functional/fma_loop_functional_test.cpp
  functional/scaling_functional_test.cpp
  functional/shared_sync_functional_test.cpp
  functional/softmax_shared_functional_test.cpp
  functional/representative_kernels_functional_test.cpp
  functional/advanced_syntax_functional_test.cpp
  functional/local_bitwise_functional_test.cpp
  functional/builtin_scalarbit_functional_test.cpp
  functional/divrem_functional_test.cpp
  functional/three_dimensional_functional_test.cpp
  functional/two_dimensional_functional_test.cpp
  functional/transpose_shared_functional_test.cpp
  cycle/cycle_smoke_test.cpp
  cycle/async_memory_cycle_test.cpp
  cycle/shared_barrier_cycle_test.cpp
  cycle/private_memory_cycle_test.cpp
  cycle/constant_memory_cycle_test.cpp
  cycle/cache_cycle_test.cpp
  cycle/shared_bank_conflict_cycle_test.cpp
  cycle/fma_loop_cycle_test.cpp
  cycle/scaling_cycle_test.cpp
  cycle/shared_sync_cycle_test.cpp
  cycle/softmax_shared_cycle_test.cpp
  cycle/representative_kernels_cycle_test.cpp
  cycle/advanced_syntax_cycle_test.cpp
  cycle/local_bitwise_cycle_test.cpp
  cycle/builtin_scalarbit_cycle_test.cpp
  cycle/divrem_cycle_test.cpp
  cycle/three_dimensional_cycle_test.cpp
  cycle/two_dimensional_cycle_test.cpp
  cycle/transpose_shared_cycle_test.cpp
  # loader/*
  loader/asm_parser_test.cpp
  loader/asm_module_integration_test.cpp
  loader/amdgpu_obj_loader_test.cpp
  loader/device_image_loader_test.cpp
  loader/device_segment_image_test.cpp
  loader/gcn_text_parser_test.cpp
  loader/program_lowering_test.cpp
  loader/amdgpu_code_object_decoder_test.cpp
  loader/executable_image_io_test.cpp
  loader/program_bundle_io_test.cpp
  loader/program_object_launch_test.cpp

  # program/*
  program/program_source_reader_test.cpp
  program/program_object_types_test.cpp

  # runtime/*
  runtime/kernel_metadata_test.cpp
  runtime/kernarg_packer_test.cpp
  runtime/hipcc_parallel_execution_test.cpp
  runtime/raw_code_object_launch_test.cpp
  runtime/model_runtime_test.cpp
  runtime/runtime_naming_test.cpp
  runtime/compatibility_alias_test.cpp
  runtime/debug_info_io_test.cpp
  runtime/cycle_timeline_test.cpp
  runtime/parallel_execution_mode_test.cpp
  runtime/hip_interposer_state_test.cpp
  runtime/hip_cts_test.cpp
  runtime/hip_feature_cts_test.cpp
  runtime/trace_test.cpp
  runtime/hip_runtime_test.cpp
  runtime/execution_stats_test.cpp
  instruction/instruction_naming_test.cpp
)

target_link_libraries(gpu_model_tests
  PRIVATE
    gpu_model
    GTest::gtest_main
)

gpu_model_enable_warnings(gpu_model_tests)
