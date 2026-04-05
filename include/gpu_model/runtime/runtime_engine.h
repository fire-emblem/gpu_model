#pragma once

// Compatibility shim: new code should include `exec_engine.h` and use
// `gpu_model::ExecEngine`. This header remains only to avoid breaking
// older includes while the rename rolls through the tree.

#include "gpu_model/runtime/exec_engine.h"
