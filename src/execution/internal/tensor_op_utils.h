// Bridge header — migrated to instruction layer.
// - IsTensorMnemonic moved to instruction/isa/tensor_isa_info.h (Layer 1, pure ISA classification)
// - TensorResultStoragePolicy and WriteTensorResult* moved to
//   instruction/semantics/internal/tensor_result_writer.h (Layer 3, semantic helpers)
//
// This file provides backward compatibility for existing includes.
// New code should include the appropriate header directly.
#pragma once

#include "instruction/isa/tensor_isa_info.h"
#include "instruction/semantics/internal/tensor_result_writer.h"
