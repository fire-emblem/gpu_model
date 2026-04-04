#include "gpu_model/debug/trace_event_view.h"

#include <string_view>

namespace gpu_model {

namespace {

TraceBarrierKind LegacyBarrierKindFromEvent(const TraceEvent& event) {
  if (event.kind != TraceEventKind::Barrier) {
    return TraceBarrierKind::None;
  }
  if (event.message == "wave") {
    return TraceBarrierKind::Wave;
  }
  if (event.message == "arrive") {
    return TraceBarrierKind::Arrive;
  }
  if (event.message == "release") {
    return TraceBarrierKind::Release;
  }
  return TraceBarrierKind::None;
}

TraceArriveKind LegacyArriveKindFromEvent(const TraceEvent& event) {
  if (event.kind != TraceEventKind::Arrive) {
    return TraceArriveKind::None;
  }
  if (event.message == "load_arrive") {
    return TraceArriveKind::Load;
  }
  if (event.message == "store_arrive") {
    return TraceArriveKind::Store;
  }
  if (event.message == "shared_arrive") {
    return TraceArriveKind::Shared;
  }
  if (event.message == "private_arrive") {
    return TraceArriveKind::Private;
  }
  if (event.message == "scalar_buffer_arrive") {
    return TraceArriveKind::ScalarBuffer;
  }
  return TraceArriveKind::None;
}

TraceLifecycleStage LegacyLifecycleStageFromEvent(const TraceEvent& event) {
  if (event.kind == TraceEventKind::WaveLaunch) {
    return TraceLifecycleStage::Launch;
  }
  if (event.kind == TraceEventKind::WaveExit &&
      (event.message == "wave_end" || event.message == "exit")) {
    return TraceLifecycleStage::Exit;
  }
  return TraceLifecycleStage::None;
}

std::string LegacyDisplayName(const TraceEvent& event) {
  if (event.kind == TraceEventKind::WaveStep) {
    constexpr std::string_view kOpPrefix = "op=";
    const std::string_view message = event.message;
    const size_t op_pos = message.find(kOpPrefix);
    if (op_pos != std::string_view::npos) {
      const size_t value_begin = op_pos + kOpPrefix.size();
      const size_t value_end = message.find_first_of(" \n", value_begin);
      return std::string(message.substr(value_begin, value_end - value_begin));
    }
  }
  return event.message;
}

std::string CanonicalNameFromLifecycle(TraceLifecycleStage stage) {
  switch (stage) {
    case TraceLifecycleStage::Launch:
      return "wave_launch";
    case TraceLifecycleStage::Exit:
      return "wave_exit";
    case TraceLifecycleStage::None:
      break;
  }
  return {};
}

std::string CanonicalNameFromBarrier(TraceBarrierKind kind) {
  switch (kind) {
    case TraceBarrierKind::Wave:
      return "barrier_wave";
    case TraceBarrierKind::Arrive:
      return "barrier_arrive";
    case TraceBarrierKind::Release:
      return "barrier_release";
    case TraceBarrierKind::None:
      break;
  }
  return {};
}

std::string CanonicalNameFromArrive(TraceArriveKind kind) {
  switch (kind) {
    case TraceArriveKind::Load:
      return "load_arrive";
    case TraceArriveKind::Store:
      return "store_arrive";
    case TraceArriveKind::Shared:
      return "shared_arrive";
    case TraceArriveKind::Private:
      return "private_arrive";
    case TraceArriveKind::ScalarBuffer:
      return "scalar_buffer_arrive";
    case TraceArriveKind::None:
      break;
  }
  return {};
}

std::string CanonicalNameFromStall(TraceStallReason reason) {
  const std::string_view reason_name = TraceStallReasonName(reason);
  if (reason_name.empty()) {
    return {};
  }
  return "stall_" + std::string(reason_name);
}

std::string KindCategory(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::Launch:
    case TraceEventKind::BlockPlaced:
    case TraceEventKind::BlockLaunch:
      return "runtime";
    case TraceEventKind::Barrier:
      return "barrier";
    case TraceEventKind::Stall:
      return "stall";
    case TraceEventKind::Arrive:
      return "arrive";
    case TraceEventKind::WaveStep:
    case TraceEventKind::Commit:
    case TraceEventKind::MemoryAccess:
    case TraceEventKind::ExecMaskUpdate:
      return "instruction";
    case TraceEventKind::WaveLaunch:
    case TraceEventKind::WaveExit:
      return "wave";
    case TraceEventKind::WaveStats:
      return "stats";
  }
  return "event";
}

}  // namespace

TraceEventView MakeTraceEventView(const TraceEvent& event) {
  TraceEventView view{
      .kind = event.kind,
      .cycle = event.cycle,
      .dpc_id = event.dpc_id,
      .ap_id = event.ap_id,
      .peu_id = event.peu_id,
      .slot_id = event.slot_id,
      .slot_model_kind = TraceEffectiveSlotModelKind(event),
      .block_id = event.block_id,
      .wave_id = event.wave_id,
      .pc = event.pc,
      .stall_reason = TraceEffectiveStallReason(event),
      .barrier_kind = event.barrier_kind,
      .arrive_kind = event.arrive_kind,
      .lifecycle_stage = event.lifecycle_stage,
      .canonical_name = {},
      .display_name = event.display_name,
      .category = KindCategory(event.kind),
      .compatibility_message = event.message,
      .used_legacy_fallback = false,
  };

  if (view.barrier_kind == TraceBarrierKind::None) {
    view.barrier_kind = LegacyBarrierKindFromEvent(event);
    view.used_legacy_fallback = view.used_legacy_fallback || view.barrier_kind != TraceBarrierKind::None;
  }
  if (view.arrive_kind == TraceArriveKind::None) {
    view.arrive_kind = LegacyArriveKindFromEvent(event);
    view.used_legacy_fallback = view.used_legacy_fallback || view.arrive_kind != TraceArriveKind::None;
  }
  if (view.lifecycle_stage == TraceLifecycleStage::None) {
    view.lifecycle_stage = LegacyLifecycleStageFromEvent(event);
    view.used_legacy_fallback =
        view.used_legacy_fallback || view.lifecycle_stage != TraceLifecycleStage::None;
  }

  if (event.kind == TraceEventKind::Barrier && view.barrier_kind != TraceBarrierKind::None) {
    view.canonical_name = CanonicalNameFromBarrier(view.barrier_kind);
  } else if (event.kind == TraceEventKind::Arrive && view.arrive_kind != TraceArriveKind::None) {
    view.canonical_name = CanonicalNameFromArrive(view.arrive_kind);
  } else if ((event.kind == TraceEventKind::WaveLaunch || event.kind == TraceEventKind::WaveExit) &&
             view.lifecycle_stage != TraceLifecycleStage::None) {
    view.canonical_name = CanonicalNameFromLifecycle(view.lifecycle_stage);
  } else if (event.kind == TraceEventKind::Stall && view.stall_reason != TraceStallReason::None) {
    view.canonical_name = CanonicalNameFromStall(view.stall_reason);
  } else if (!event.display_name.empty()) {
    view.canonical_name = event.display_name;
  } else {
    view.canonical_name = LegacyDisplayName(event);
    view.used_legacy_fallback = view.used_legacy_fallback || !view.canonical_name.empty();
  }

  if (view.display_name.empty()) {
    if (event.kind == TraceEventKind::Barrier && view.barrier_kind != TraceBarrierKind::None) {
      view.display_name = std::string(TraceBarrierKindName(view.barrier_kind));
    } else if (event.kind == TraceEventKind::Arrive && view.arrive_kind != TraceArriveKind::None) {
      view.display_name = std::string(TraceArriveKindName(view.arrive_kind));
    } else if ((event.kind == TraceEventKind::WaveLaunch || event.kind == TraceEventKind::WaveExit) &&
               view.lifecycle_stage != TraceLifecycleStage::None) {
      view.display_name = std::string(TraceLifecycleStageName(view.lifecycle_stage));
    } else if (event.kind == TraceEventKind::Stall && !view.canonical_name.empty()) {
      view.display_name = view.canonical_name;
    } else {
      view.display_name = LegacyDisplayName(event);
    }
    view.used_legacy_fallback = view.used_legacy_fallback || !view.display_name.empty();
  }

  return view;
}

}  // namespace gpu_model
