# Examples

面向用户的可执行 HIP kernel 例子，按难度编号 01-11。

## 执行模式

每个例子默认运行三种模式：

| 模式 | 全称 | 说明 |
|------|------|------|
| `st` | SingleThreaded | 单线程功能执行，确定性语义参考 |
| `mt` | MultiThreaded | 多线程功能执行，Marl fiber 并行 |
| `cycle` | Cycle | Naive cycle 模型，带时间线估算 |

三种模式 host 侧校验结果应完全一致。

## 例子列表

| 编号 | 例子 | 作用 | 验证重点 |
|------|------|------|----------|
| 01 | [vecadd-basic](./01-vecadd-basic) | 最小端到端入门 | 基础路径接通、三种模式一致性 |
| 02 | [fma-loop](./02-fma-loop) | 循环 + 浮点累积 | 控制流、浮点路径稳定性 |
| 03 | [shared-reverse](./03-shared-reverse) | shared memory + barrier | block 内同步、shared 读写 |
| 04 | [atomic-reduction](./04-atomic-reduction) | global atomic 归约 | 原子语义、并发写正确性 |
| 05 | [softmax-reduction](./05-softmax-reduction) | 多阶段归约 | 多次 barrier + 数值计算 |
| 06 | [mma-gemm](./06-mma-gemm) | MFMA 探针 | gfx90a/mfma 能力检测 |
| 07 | [vecadd-cycle-splitting](./07-vecadd-cycle-splitting) | 写法对比 | 不同实现的 cycle 差异 |
| 08 | [conditional-multibarrier](./08-conditional-multibarrier) | 条件分支 + 多次 barrier | 合法 barrier 与条件分支共存 |
| 09 | [dynamic-shared-sum](./09-dynamic-shared-sum) | 动态 shared memory | `extern __shared__` 路径 |
| 10 | [block-reduce-sum](./10-block-reduce-sum) | 多 block 归约 | 独立 block reduction |
| 11 | [perfetto-waitcnt-slots](./11-perfetto-waitcnt-slots) | Perfetto 调试 | 空泡、slot、wave 调度可视化 |

## 阅读顺序

1. **01-05**: 基础功能验证（算术、shared、barrier、atomic、reduction）
2. **06**: MFMA 能力探针（可能标记 `unsupported_yet`）
3. **07**: 写法对比分析
4. **08-10**: 复杂同步模式
5. **11**: Trace 观察能力

## 成功标准

运行成功后：
- `stdout.txt` 包含 `... validation ok` 或 `mismatches=0`
- `launch_summary.txt` 包含 `ok=1`
- `trace.txt` / `trace.jsonl` / `timeline.perfetto.json` 非空

## 故障排查

```bash
# 1. 检查构建
cmake --build --preset dev-fast

# 2. 检查测试
./build-ninja/tests/gpu_model_tests --gtest_filter=*VecAdd*

# 3. 检查单个例子
./examples/01-vecadd-basic/run.sh
cat examples/01-vecadd-basic/results/st/stdout.txt
```

## Trace 产物

- `timeline.perfetto.json`: Chrome Trace 格式，用于可视化分析
- `trace.jsonl`: JSON Lines 格式，适合程序处理
- `trace.txt`: 纯文本格式，适合快速浏览

Slot 语义：
- `cycle`: `resident_fixed` (真实 resident slot)
- `st/mt`: `logical_unbounded` (逻辑 S* 轨道)
