# 11 Perfetto Waitcnt Slots

## 例子作用

这个例子不是通用功能样例，而是一个**专门为了 Perfetto 可视化而构造的调试样例集**。

它的目标非常明确：

- 让 timeline 上出现明显空泡
- 让多个 wave 同时可见
- 让 `wave_launch / wave_switch_away / load_arrive / wave_exit` 都能看清
- 区分 `cycle` 的 `resident_fixed` 和 `st/mt` 的 `logical_unbounded`
- 优先服务于 Perfetto 观测，而不是通用 host 业务逻辑

## 本例关注点

这个例子包含三个子 case：

- `timeline_gap`
  最适合看 `waitcnt` 导致的明显空泡
- `same_peu_slots`
  最适合看同一个 `PEU` 下多个 slot / wave 的并行存在
- `switch_away_heavy`
  最适合看调度器反复切走 wave 的节奏

同时它会生成三组模式：

- `st`
- `mt`
- `cycle`

因此总共是 9 组结果目录。

## 统一语义

当前 trace/perfetto 导出的统一槽位语义是：

- `cycle`
  使用 `resident_fixed`
  也就是同一个 `PEU` 下只展示真实 resident slot
- `st/mt`
  使用 `logical_unbounded`
  也就是只要 wave 被 dispatch 到这个 `PEU`，就给它分配一个逻辑 `S*` 轨道

这套语义同时覆盖：

- `InstructionBuilder` kernel
- `encoded_program_object` kernel

## 运行方式

```bash
./examples/11-perfetto-waitcnt-slots/run.sh
```

## 关键产物

结果目录结构：

- `results/st/timeline_gap`
- `results/st/same_peu_slots`
- `results/st/switch_away_heavy`
- `results/mt/timeline_gap`
- `results/mt/same_peu_slots`
- `results/mt/switch_away_heavy`
- `results/cycle/timeline_gap`
- `results/cycle/same_peu_slots`
- `results/cycle/switch_away_heavy`

每个 case 目录都应包含：

- `stdout.txt`
- `trace.txt`
- `trace_parts/`（仅当原始 `trace.txt` 超过 2000 行时生成）
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

根目录还会额外生成：

- `guide.txt`
- `summary.txt`

## 如何阅读产物

优先顺序：

1. `timeline.perfetto.json`
   这是 Chrome trace JSON，主要用于文本检查、grep、diff
2. `trace.jsonl`
   适合看逐事件字段
3. `trace.txt`
   适合人肉快速浏览；如果目录里同时存在 `trace_parts/`，说明这里是预览+索引，
   完整正文已经按 1800 行分片到 `trace_parts/trace_part_000.txt`、`trace_part_001.txt` ...

## 预期结果

运行成功后：

- `summary.txt` 应出现 `perfetto_waitcnt_slots_demo ok`
- 9 个 case 目录的 `stdout.txt` 都应包含 `ok=1`

具体到三个子 case：

`timeline_gap`

- 预期看到：
  - `buffer_load_dword`
  - `stall_waitcnt_global`
  - 一段没有指令切片的空白时间
  - `load_arrive`
  - 恢复后的下一条指令

按当前仓库这次重跑结果：

- `cycle/timeline_gap`
  - `total_cycles = 60`
  - 第一段明显空泡约为 `36 cycles`
  - trace 中可见 `stall_waitcnt_global`、`load_arrive`、`wave_exit`

`same_peu_slots`

- `cycle` 下预期看到：
  - `resident_fixed`
  - 多个 `PEU`
  - 每个 `PEU` 下多个固定 `S*`
- `st/mt` 下预期看到：
  - `logical_unbounded`
  - 同一个 `PEU` 下很多逻辑 `S*`
  - `wave_switch_away`
  - `wave_exit`

按当前仓库这次重跑结果：

- `st/same_peu_slots`
  - `total_cycles = 1032`
  - 可见 `4` 个 `PEU` process
  - 总 thread track 数为 `33`
  - `wave_switch_away = 194`
  - `load_arrive = 33`
  - `wave_exit = 33`
- `mt/same_peu_slots`
  - `total_cycles = 1032`
  - 可见 `4` 个 `PEU` process
  - 总 thread track 数为 `33`
  - `wave_switch_away = 193`
  - `load_arrive = 33`
  - `wave_exit = 33`

`switch_away_heavy`

- 预期看到：
  - 高频 `wave_switch_away`
  - waitcnt 事件不是主角
  - timeline 更像调度轮转图，而不是内存等待图

按当前仓库这次重跑结果：

- `cycle/switch_away_heavy`
  - `total_cycles = 524`
  - `4` 个 `PEU` process
  - 每个 `PEU` 都有 `50` 次 `wave_switch_away`
  - `load_arrive = 16`
  - `wave_exit = 16`

当前仓库已有 `summary.txt` 给出的 quick-start 顺序就是合理入口：

1. `results/guide.txt`

## 调试建议

- 如果你关心“空泡是不是足够明显”，先看 `cycle/timeline_gap`
- 如果你关心“st/mt 多个 wave 是否真的同时可见”，先看 `st/mt/same_peu_slots`
- 如果你关心“调度切换是否被稳定观测到”，先看 `cycle/switch_away_heavy`
- 看 `st/mt/same_peu_slots` 时，不要只看 thread 名。
  同名 `WAVE_SLOT_00..08` 会在不同 `PEU` 下重复出现，必须结合 `process_name`
  一起理解层级。
- `timeline_gap` 里的“空泡”并不是完全没有 trace event，而是没有 instruction slice。
  trace.txt 中仍会看到一串 `stall_waitcnt_global` 事件，这是当前导出语义的一部分。
- `same_peu_slots` 和 `switch_away_heavy` 的 `st/mt trace.txt` 可能非常长。
  当前仓库把它们收口成“短预览 + trace_parts 分片正文”，避免把几十万行裸文本直接留在结果根目录。

## 结果解读

- 这个例子通过，说明当前 trace / Perfetto 观察能力基本符合预期
- 它验证的是“能否看见空泡、slot、wave 生命周期、switch-away”，不是某个业务 kernel 的数值正确性
- 如果这里观察不到预期现象，应优先怀疑 trace 语义、事件布局或导出格式，而不是先怀疑 Perfetto 本身
