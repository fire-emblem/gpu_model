# 发现与决策

## 需求
- 先暂停继续扩张 cycle 前端建模，切回 examples 主题。
- 需要对 `examples` 下的任务做全量、分批检查。
- 当前用户明确指出：
  - `08` 的 `mt` Perfetto 显示不正确
  - `11` 编译不过
  - Perfetto 必须显示每条指令的 `4 cycle` 区间

## 研究发现
- `examples/08-conditional-multibarrier` 的 `mt` Perfetto 异常不是单一问题：
  - example 构造本身不适合展示“同一 PEU 上多 slot/多 wave 并发”
  - 真实 HIP functional encoded 路径还存在事件生产缺口，导致之前完全没有 instruction slice
- `examples/08` 的真实根因已经定位：
  - `ExecEngine` 在 `ExecutionMode::Functional + encoded_program_object` 时走 `EncodedExecEngine`
  - `EncodedExecEngine::ExecuteWave()` 原先只发 `WaveStep`，没有发 `Commit`
  - `CycleTimelineRenderer` 依赖 `WaveStep + Commit` 配对生成 instruction segment
  - 因此 functional `st/mt` 的 `timeline.perfetto.json/.pb` 只有 marker，没有 `ph:"X"` 指令区间
- 修复后已验证：
  - `08 mt` 的 `trace.jsonl` 从 `Commit=0` 变为 `Commit=872`
  - `08 mt` 的 `timeline.perfetto.json` 从 `X_count=0` 变为 `X_count=728`
  - 所有 instruction slice 的 `dur` 都是 `4`
- `examples/11-perfetto-waitcnt-slots` 编译失败根因已确认：
  - example 代码仍引用旧的 `gpu_model/runtime/runtime_engine.h` 和 `gm::RuntimeEngine`
  - 当前仓库正式接口已是 `gpu_model/runtime/exec_engine.h` 与 `gm::ExecEngine`
- `HipInterposerState` 原本只是对 `RuntimeSession` 的薄包装，已经不值得单独保留。
- `hip_interposer.cpp` 中的多数关键路径可以安全收口到 `HipRuntime`：
  - kernel 注册
  - launch config push/pop
  - memory ops
  - device query
  - stream/event
  - last error
  - executable launch
- 删除 `HipInterposerState` 后，`HipInterposerStateTest.*:InterposerCTS/*:InterposerFeatureCTS/*` 全量通过。
- 轻量门禁 smoke 覆盖足以快速发现架构层回归：
  - `RuntimeNamingTest.*`
  - `RuntimeProgramCompatibilityAliasTest.*`
  - `HipInterposerStateTest.RunsHipHostExecutableThroughLdPreloadInterposer`
  - `HipRuntimeTest.LaunchKernelCanReadMaterializedDataPool`
  - `ModelRuntimeCoreTest.SimulatesMallocMemcpyLaunchAndSynchronizeFlow`
  - `TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents`
- `ExecEngine` 渐进迁移可安全分三步完成：
  - 先引入公开名字和 shim
  - 再切 high-level 头文件与测试使用
  - 最后切物理实现文件与底层类名
- 当前 docs 命中分析显示：
  - 对外文档基本已收口
  - 剩余旧名主要集中在 `docs/superpowers/plans`、`docs/superpowers/specs`、`docs/plans`
- example runner 默认写 `.cache/example-results` 会让用户很容易误读旧 `results/` 快照
- 当前已改回：
  - example 结果直接写回各自目录下的 `results/`
  - 不再保留 `.cache/example-results` 这条默认路径
- 当前已停止生成 `timeline.perfetto.pb`
  - 正式保留的时间线产物只有 `timeline.perfetto.json`
  - 现有手写 `.pb` exporter 只保留内部代码与测试，不再对用户暴露为正式产物
- `Functional` 时间轴当前已从“trace 事件计数器”收口为“执行状态推进时间”：
  - 纯 scalar `100` 指令在 `st/mt/cycle` 下都已由 focused test 锁定为相邻 issue 差 `4 cycle`
  - `Functional` 的 dense global load overlap 与 implicit drain 也已由 focused test 锁定
- `Functional` 的恢复语义已明确：
  - `st`：`arrive_resume` 后，消费者指令在下一 issue quantum 起点发出
  - `mt`：共用同一 modeled time 规则，但保留 runnable wave 竞争，不保证强同步恢复
- `Cycle` 的结果型证明当前已形成闭环：
  - front-end latency 推进不依赖 trace
  - resident/standby/promote/backfill 行为通过
  - dense global load overlap 与 `endk` 隐式 drain 通过
  - `ready != selected != issue` 已由结果型测试证明

## 技术决策
| 决策 | 理由 |
|------|------|
| 删除 `HipInterposerState` | 已无主路径使用点，只会制造额外层级 |
| `hip_interposer.cpp` 归属到 `HipRuntime` 语义 | 满足“HIP 兼容层就是对外 C 接口”目标 |
| `RuntimeEngine` 渐进改名为 `ExecEngine` | 名字更准确，且可通过兼容别名平滑迁移 |
| 保留 `runtime_engine.h` 兼容 shim | 降低一次性全仓重命名风险 |
| 轻量 pre-push + 手动 full gate | 平衡开发效率与验证覆盖 |
| 历史存档文档中的旧名也开始统一替换 | 当前主线已经稳定，继续保留旧名的成本高于保留原貌的收益 |
| example 结果重新固定写回 `results/` | 避免默认结果与仓库快照分叉，降低误读成本 |
| 停止生成 `timeline.perfetto.pb` artifact | 当前 `.pb` 不是可靠的官方 Perfetto 兼容产物，对用户暴露会造成误导 |
| 当前所有模型 trace 的 `cycle` 都明确定义为模型计数时间，不是物理真实执行时间戳 | 避免用 `st/mt/cycle` 的 trace 时间误判 AP 并行性、resume 时点和真实性能结论 |
| 提供 `GPU_MODEL_DISABLE_TRACE=1` 全局开关 | 便于在快速推进模型时彻底关闭 trace 干扰，并验证非 trace 模块语义不受影响 |
| `Functional` 恢复语义采用“下一 issue quantum 起点消费”，`Cycle` 只表达 ready 不保证 issue 时刻 | 让 `st` 成为确定性语义参考，同时保留 `mt/cycle` 的调度竞争差异 |
| cycle 业务逻辑只放在 engine / state machine，trace 只做消费序列化 | 保证行为事实和展示分层清晰 |
| 当前对外架构文档不再把 `hip_interposer` 当模块名 | 文件名可以保留历史字样，但架构语义必须归到 `HipRuntime` |
| 暂不系统清理 docs 存档中的历史旧名 | 历史计划/spec 属于存档信息，优先保证主代码和关键文档收口 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| preload 测试在 ASan build 下失败 | 将 `libasan` 优先加入 `LD_PRELOAD` |
| `HipRuntime` 兼容态内存与 `ModelRuntime` 自带 memory 混淆 | 为 `HipRuntime` 显式提供 `compatibility_memory()` |
| `ExecEngine` 迁移初期 include/类型替换容易误伤 | 先引入别名与 shim，再分批替换高层使用点 |

## 资源
- 关键代码：
  - `src/runtime/hip_interposer.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `src/runtime/core/model_runtime.cpp`
  - `src/runtime/exec_engine.cpp`
- 关键脚本：
  - `scripts/run_push_gate.sh`
  - `scripts/run_push_gate_light.sh`
  - `.githooks/pre-push`

## 视觉/浏览器发现
- 本阶段无浏览器/视觉外部信息输入。
- 当前 cycle model 的正确定义仍然保持：
  - 唯一 cycle 模式，不区分 `st/mt`
  - 通过资源、时序、issue policy、slot 约束来表达不同硬件行为
  - 不是通过宿主执行模式表达
- 但本轮优先级已切到 examples/Perfetto 正确性，而不是继续扩 cycle 范围
