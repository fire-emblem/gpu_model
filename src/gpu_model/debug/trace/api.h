#pragma once

// Stable umbrella header for the debug/trace API.
//
// Consumers outside the debug implementation should prefer this header over
// reaching into debug/internal/. Files under debug/internal/ are implementation
// details and are not part of the stable module contract.

#include "gpu_model/debug/recorder/export.h"
#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/replay/replayer.h"
#include "gpu_model/debug/timeline/cycle_timeline.h"
#include "gpu_model/debug/trace/artifact_recorder.h"
#include "gpu_model/debug/trace/event.h"
#include "gpu_model/debug/trace/event_export.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/event_view.h"
#include "gpu_model/debug/trace/sink.h"
