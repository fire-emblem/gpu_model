# Examples

这里放面向客户的可执行例子。

## 约束

- 每个例子一个目录
- 每个例子都是真实 HIP 源码
- 每个例子都能先通过 `hipcc` 生成 `.out`
- 每个目录都有独立 `run.sh`
- 目录按由简到难编号
- 当前特性尚未完全接通的部分，在目录 `README.md` 里明确标记

## 执行模式说明

每个例子默认运行三种执行模式：

| 模式 | 全称 | 说明 |
|------|------|------|
| `st` | SingleThreaded | 单线程功能执行，确定性语义参考模型 |
| `mt` | MultiThreaded | 多线程功能执行，使用 Marl fiber 并行 |
| `cycle` | Cycle | Naive cycle 级别模型，带时间线开销估算 |

三种模式的 host 侧校验结果应完全一致，区别在于执行调度策略和 trace 时间语义。

## Trace 产物说明

- 凡是通过 `TraceArtifactRecorder` 落盘 trace 的例子，都会产出 `timeline.perfetto.json`
- `timeline.perfetto.json` 是当前唯一正式支持的时间线产物，适合文本检查、grep、回归 diff 和后续格式转换
- Slot 语义：
  - `cycle` 使用 `resident_fixed`：只展示真实 resident slot
  - `st/mt` 使用 `logical_unbounded`：同一个 `PEU` 上有多少 wave 就展示多少逻辑 `S*` 轨道

## 例子列表

| 编号 | 例子 | 作用 | 验证重点 |
|------|------|------|----------|
| 01 | [vecadd-basic](./01-vecadd-basic) | 最小端到端入门 | 基础路径接通、st/mt/cycle 一致性 |
| 02 | [fma-loop](./02-fma-loop) | 循环 + 浮点累积 | 控制流、浮点路径稳定性 |
| 03 | [shared-reverse](./03-shared-reverse) | shared memory + barrier | block 内同步、shared 读写 |
| 04 | [atomic-reduction](./04-atomic-reduction) | global atomic 归约 | 原子语义、并发写正确性 |
| 05 | [softmax-reduction](./05-softmax-reduction) | 多阶段归约 | 多次 barrier、shared + 数值计算 |
| 06 | [mma-gemm](./06-mma-gemm) | MFMA 探针 | gfx90a/mfma 能力检测 |
| 07 | [vecadd-cycle-splitting](./07-vecadd-cycle-splitting) | 写法对比 | 不同实现方式的 cycle 差异 |
| 08 | [conditional-multibarrier](./08-conditional-multibarrier) | 条件分支 + 多次 barrier | 合法 barrier 与条件分支共存 |
| 09 | [dynamic-shared-sum](./09-dynamic-shared-sum) | 动态 shared memory | `extern __shared__` 路径 |
| 10 | [block-reduce-sum](./10-block-reduce-sum) | 多 block 归约 | 独立 block reduction |
| 11 | [perfetto-waitcnt-slots](./11-perfetto-waitcnt-slots) | Perfetto 调试样例 | 空泡、slot、wave 调度可视化 |

## 阅读顺序建议

**基础功能验证（01-05）**
- 先看算术、shared、barrier、atomic、reduction 基础功能是否正确
- `01` 是所有例子的前置基线

**能力探针（06）**
- 看 MMA/MFMA 能力是否接通
- 注意：此例可能因环境不支持而标记 `unsupported_yet`

**写法对比分析（07）**
- 看同一语义的不同写法在 cycle 上有什么差异
- 理解不同实现策略的成本影响

**复杂同步模式（08-10）**
- 看更复杂同步与归约模式
- `08`: 多次 barrier + 条件分支
- `09`: 动态 shared
- `10`: 多 block reduction

**Trace 观察能力（11）**
- 专门看 trace/perfetto 观察能力
- 空泡、slot 语义、wave 调度可视化

关于 `results/`：

- `run.sh` 会直接把结果写回当前 example 目录下的 `results/`
- README 中描述的“预期结果”以当前 `run.sh` 重新生成的产物为准

关于并行运行：

- `examples/common.sh` 里的构建步骤现在已经带文件锁
- 可以并行启动多个 `run.sh`
- 它们会串行进入同一个 build 目录的 `cmake --build`，避免破坏 `build-ninja`
  的 `.ninja_log` / `.ninja_deps`
