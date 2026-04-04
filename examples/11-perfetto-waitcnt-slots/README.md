# 11 Perfetto Waitcnt Slots

这个例子专门给 Perfetto timeline 看三类现象：

- `timeline_gap`
  用最小 waitcnt kernel 生成明显空泡。时间轴上可以看到指令切片、等待期间空白、`load_arrive` 和后续恢复。
- `same_peu_slots`
  用同一 `PEU` 上堆很多 wave 的 kernel 生成多个 slot 轨道。更适合看 `S0/S1/S2/...`、wave start/end、`wave_switch_away`，以及多个 wave 在逻辑槽位上的并行存在。
- `switch_away_heavy`
  用几乎全是 ALU 的多 wave kernel 放大调度切换现象。更适合专门看 `wave_switch_away` 的轮转节奏。

这套展示约定现在不是只对这个例子生效，而是当前 trace/perfetto 导出的统一语义：

- `cycle`
  使用 `resident_fixed`。同一个 `PEU` 下只展示真实 resident slot。
- `st/mt`
  使用 `logical_unbounded`。同一个 `PEU` 下有多少 wave dispatch 到这里，就展示多少个逻辑 `S*` 轨道。
- 这套语义同时覆盖 `InstructionBuilder` kernel 和 `encoded_program_object` kernel。

当前每个目录都会同时导出两种 Perfetto 文件：

- `timeline.perfetto.pb`
  native Perfetto trace。优先打开这个文件，可以看到真正的 `TrackDescriptor(parent_uuid)`
  层级，也就是 `Device -> DPC -> AP -> PEU -> Slot` 的可折叠父子 track。
- `timeline.perfetto.json`
  Chrome JSON trace。保留它主要是为了文本检查、回归测试和快速 diff。

其中 `timeline.perfetto.json` 使用的是 Chrome JSON trace 格式。
这意味着 Perfetto 原生只能折叠 `process/thread` 两级，不能直接表达 `Device -> DPC -> AP -> PEU -> Slot`
这种多级 nested track。当前实现采用的可视化折中是：

- process 名称使用扁平路径数字标签，例如 `D0/A0/P0`
- thread 名称使用 slot 数字标签，例如 `S0`
- metadata 里额外写出 `hierarchy_levels`、`track_layout` 和 `perfetto_format`

所以当前设计的主要限制不是例子构造，而是 `timeline.perfetto.json` 这个导出格式本身。
如果后续要让每一级都在 Perfetto 里单独折叠，需要新增 native Perfetto `TrackDescriptor(parent_uuid)` 导出。

输出目录：

- `results/st/timeline_gap`
- `results/st/same_peu_slots`
- `results/st/switch_away_heavy`
- `results/mt/timeline_gap`
- `results/mt/same_peu_slots`
- `results/mt/switch_away_heavy`
- `results/cycle/timeline_gap`
- `results/cycle/same_peu_slots`
- `results/cycle/switch_away_heavy`

每个目录都会生成：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.pb`
- `timeline.perfetto.json`
- `launch_summary.txt`

结果根目录还会额外生成：

- `guide.txt`
  直接告诉你先看哪个 case、每个 case 最适合观察什么现象。
- `summary.txt`
  由 `run.sh` 生成，除了运行日志，还会附带 quick-start 入口。

运行：

```bash
./examples/11-perfetto-waitcnt-slots/run.sh
```

建议观察：

- 优先打开 `timeline.perfetto.pb`:
  这是 native Perfetto 层级版，应该可以直接折叠 `Device/D0/A0/P0/S0` 这些层级。
- 需要看文本结构或回归 diff 时再看 `timeline.perfetto.json`:
  这个版本保留了 `slot_model`、数字路径标签和便于 grep 的 JSON 结构。

- `timeline_gap`:
  看 `buffer_load_dword` 后的等待空白、`stall_waitcnt_global` / `load_arrive`、以及恢复后的下一条指令。
- `same_peu_slots`:
  看同一个 `PEU` 的 process 路径 `D*/A*/P*` 下多个 `S*` 轨道。
  这个 case 现在更适合作为“多逻辑槽位 + wave_switch_away”的观察入口。
  `wave_switch_away` 表示该 wave 因调度切换被让出执行；等待空泡、`stall_waitcnt_global` 和 `load_arrive` 更适合去看 `timeline_gap`。
  `st/mt` 会显示 `logical_unbounded` 槽位语义，`cycle` 会显示 `resident_fixed`。
  如果你打开的是别的 trace dump，而不是这个例子，观察规则也是一样的。
- `switch_away_heavy`:
  优先看 `cycle/switch_away_heavy`。
  这个 case 里 `wave_switch_away` 会比 waitcnt 事件更密集、更主导，适合单独观察多个 wave 在 slot 之间轮转推进。
