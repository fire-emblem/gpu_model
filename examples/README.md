# Examples

面向用户的可执行 HIP kernel 例子，按难度编号组织。

## 执行模式

默认规则：

- 非对比型 example 默认只跑 `mt`
- 对比型 / 可视化型 example 显式保留 `st` / `mt` / `cycle`
- 专项例子可以只跑单一模式，例如 `test-sm-copy` 只跑 `cycle`
- 默认启用 `tools/hipcc_cache.sh` 复用 example 编译结果，可用 `GPU_MODEL_USE_HIPCC_CACHE=0` 关闭

模式说明：

| 模式 | 全称 | 说明 |
|------|------|------|
| `st` | SingleThreaded | 单线程功能执行，确定性语义参考 |
| `mt` | MultiThreaded | 多线程功能执行，Marl fiber 并行 |
| `cycle` | Cycle | Naive cycle 模型，带时间线估算 |

## 例子列表

| 编号 | 例子 | 作用 | 验证重点 |
|------|------|------|----------|
| 01 | [vecadd-basic](./01-vecadd-basic) | 最小端到端入门 | 基础路径接通、默认 `mt` |
| 02 | [fma-loop](./02-fma-loop) | 循环 + 浮点累积 | 控制流、浮点路径稳定性、默认 `mt` |
| 03 | [shared-reverse](./03-shared-reverse) | shared memory + barrier | block 内同步、shared 读写、默认 `mt` |
| 04 | [atomic-reduction](./04-atomic-reduction) | global atomic 归约 | 原子语义、并发写正确性、默认 `mt` |
| 05 | [softmax-reduction](./05-softmax-reduction) | 多阶段归约 | 多次 barrier + 数值计算、默认 `mt` |
| 06 | [mma-gemm](./06-mma-gemm) | MFMA 探针 | gfx90a/mfma 能力检测、默认 `mt` |
| 07 | [vecadd-cycle-splitting](./07-vecadd-cycle-splitting) | 写法对比 | 保留 `st/mt/cycle` 对比 |
| 08 | [conditional-multibarrier](./08-conditional-multibarrier) | 条件分支 + 多次 barrier | 合法 barrier 与条件分支共存、默认 `mt` |
| 09 | [dynamic-shared-sum](./09-dynamic-shared-sum) | 动态 shared memory | `extern __shared__` 路径、默认 `mt` |
| 10 | [block-reduce-sum](./10-block-reduce-sum) | 多 block 归约 | 独立 block reduction、默认 `mt` |
| 11 | [perfetto-waitcnt-slots](./11-perfetto-waitcnt-slots) | Perfetto 调试 | 保留 `st/mt/cycle` 可视化 |
| 12 | [schedule-strategy-comparison](./12-schedule-strategy-comparison) | 调度策略对比 | 保留 `st/mt/cycle` 对比 |
| 13 | [algorithm-comparison](./13-algorithm-comparison) | 算法对比 | 保留 `st/mt/cycle` 对比 |
| test-sm-copy | [test-sm-copy](./test-sm-copy) | shared-memory copy 专项 | 只跑 `cycle` |

## 阅读顺序

1. **01-05**: 基础功能验证，默认 `mt`
2. **06**: MFMA 能力探针，默认 `mt`，可能标记 `unsupported_yet`
3. **07**: 写法对比分析，保留 `st/mt/cycle`
4. **08-10**: 复杂同步模式，默认 `mt`
5. **11-13**: Trace / 调度 / 算法对比，保留 `st/mt/cycle`

## 成功标准

运行成功后：
- `stdout.txt` 包含 `... validation ok` 或 `mismatches=0`
- `trace.txt` [SUMMARY] 部分包含 `kernel_status=PASS` 和 `launch_index=`
- `trace.txt` / `trace.jsonl` / `timeline.perfetto.json` 非空

对非对比型 example，上述产物默认只在 `results/mt/` 下生成。

## 故障排查

```bash
# 1. 检查构建
cmake --build --preset dev-fast

# 2. 检查测试
./build-ninja/tests/gpu_model_tests --gtest_filter=*VecAdd*

# 3. 检查单个例子
./examples/01-vecadd-basic/run.sh
cat examples/01-vecadd-basic/results/mt/stdout.txt

# 4. 如需禁用 example 的 hipcc 缓存
GPU_MODEL_USE_HIPCC_CACHE=0 ./examples/01-vecadd-basic/run.sh
```

## Trace 产物

- `timeline.perfetto.json`: Chrome Trace 格式，用于可视化分析
- `trace.jsonl`: JSON Lines 格式，适合程序处理
- `trace.txt`: 纯文本格式，适合快速浏览

Slot 语义：
- `cycle`: `resident_fixed` (真实 resident slot)
- `st/mt`: `logical_unbounded` (逻辑 S* 轨道)
