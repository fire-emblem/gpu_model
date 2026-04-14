#pragma once

// Stable umbrella header for the debug/trace API.
//
// Consumers outside the debug implementation should prefer this header over
// reaching into debug/internal/. Files under debug/internal/ are implementation
// details and are not part of the stable module contract.

#include "debug/recorder/export.h"
#include "debug/recorder/recorder.h"
#include "debug/timeline/cycle_timeline.h"
#include "debug/trace/artifact_recorder.h"
#include "debug/trace/event.h"
#include "debug/trace/event_export.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/event_view.h"
#include "debug/trace/sink.h"
