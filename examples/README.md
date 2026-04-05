# Examples

这里放面向客户的可执行例子。

约束：

- 每个例子一个目录
- 每个例子都是真实 HIP 源码
- 每个例子都能先通过 `hipcc` 生成 `.out`
- 每个目录都有独立 `run.sh`
- 目录按由简到难编号
- 当前特性尚未完全接通的部分，在目录 `README.md` 里明确标记

补充说明：

- 凡是通过 `TraceArtifactRecorder` 落盘 trace 的例子，都会产出 `timeline.perfetto.json`
- `timeline.perfetto.json` 是当前唯一正式支持的时间线产物，适合文本检查、grep、回归 diff 和后续格式转换
- 目前统一的 slot 语义是：
  `cycle` 使用 `resident_fixed`，只展示真实 resident slot；
  `st/mt` 使用 `logical_unbounded`，同一个 `PEU` 上有多少 wave 就展示多少逻辑 `S*` 轨道

当前目标例子：

1. [01-vecadd-basic](./01-vecadd-basic)
2. [02-fma-loop](./02-fma-loop)
3. [03-shared-reverse](./03-shared-reverse)
4. [04-atomic-reduction](./04-atomic-reduction)
5. [05-softmax-reduction](./05-softmax-reduction)
6. [06-mma-gemm](./06-mma-gemm)
7. [07-vecadd-cycle-splitting](./07-vecadd-cycle-splitting)
8. [08-conditional-multibarrier](./08-conditional-multibarrier)
9. [09-dynamic-shared-sum](./09-dynamic-shared-sum)
10. [10-block-reduce-sum](./10-block-reduce-sum)
11. [11-perfetto-waitcnt-slots](./11-perfetto-waitcnt-slots)

阅读建议：

- `01-05`
  先看基础功能是否正确，包括算术、shared、barrier、atomic、reduction
- `06`
  看 MMA / MFMA 能力是否接通
- `07`
  看同一语义的不同写法在 cycle 上有什么差异
- `08-10`
  看更复杂同步与归约模式
- `11`
  专门看 trace / perfetto 观察能力、空泡、slot 语义和 wave 调度可视化

关于 `results/`：

- `run.sh` 会直接把结果写回当前 example 目录下的 `results/`
- README 中描述的“预期结果”以当前 `run.sh` 重新生成的产物为准

关于并行运行：

- `examples/common.sh` 里的构建步骤现在已经带文件锁
- 可以并行启动多个 `run.sh`
- 它们会串行进入同一个 build 目录的 `cmake --build`，避免破坏 `build-ninja`
  的 `.ninja_log` / `.ninja_deps`
