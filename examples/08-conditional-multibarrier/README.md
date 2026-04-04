# 08 Conditional Multibarrier

## 例子作用

这个例子展示一个**合法的**多次 barrier 样例，重点不是“barrier 多”，而是“带条件分支时 barrier 仍然合法”。

它用于验证：

- 条件计算路径变化是否会破坏 barrier 语义
- 所有线程是否仍按一致顺序经过每一次 `__syncthreads()`
- 多 block 下的多次 barrier 是否稳定

## 本例子的关注点

- 条件分支与 barrier 能否合法共存
- 三次 barrier 是否都被所有线程按相同顺序经过
- block 内多阶段 shared 数据变换是否正确
- 不同 block 是否独立执行且结果一致

## 程序行为

这个 kernel 会做三轮带 barrier 的 block 内计算：

1. 初始写入 shared memory
2. 第一轮条件计算后 barrier
3. 第二轮条件混合后 barrier
4. 最终把结果写回 global memory

关键点在于：

- 条件分支只改变计算内容
- **不会让任何线程跳过 barrier**

因此这是一个“条件分支存在，但 barrier 仍合法”的例子。

## 运行方式

```bash
./examples/08-conditional-multibarrier/run.sh
```

脚本会：

1. 编译 [conditional_multibarrier.hip](/data/gpu_model/examples/08-conditional-multibarrier/conditional_multibarrier.hip)
2. 分别以 `st` / `mt` / `cycle` 运行
3. 检查 `stdout.txt` 中是否包含 `conditional_multibarrier mismatches=0`

## 关键产物

运行成功后应看到：

- `results/conditional_multibarrier.out`
- `results/st`
- `results/mt`
- `results/cycle`

每个模式目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

按当前主线重跑后，还可能看到：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准：

- 三个模式的 `stdout.txt` 都包含 `conditional_multibarrier mismatches=0`
- 三个模式的 `launch_summary.txt` 中都应有 `ok=1`
- trace 中应能看到多次 barrier 相关事件
- 不应出现因为条件分支导致的 barrier 死锁、缺失或异常退出

## 建议观察项

这个例子建议重点观察：

- 多次 barrier 是否形成完整闭合的阶段边界
- 条件分支是否只影响计算结果，不影响 barrier 到达顺序
- 不同 block 是否都完整执行了所有阶段

如果你想专门看“非法 barrier”类问题，这个例子不合适；它展示的是合法写法。

## 调试入口

建议按顺序排查：

1. `results/<mode>/stdout.txt`
   先看是否 `mismatches=0`
2. `results/<mode>/trace.txt`
   看三次 barrier 相关事件是否都出现
3. `results/<mode>/launch_summary.txt`
   确认 kernel 是否正常结束
4. `timeline.perfetto.*`
   需要按时间线看 barrier 阶段时再打开

## 结果解读

- 这个例子通过，说明“条件分支存在但 barrier 合法”的模式已基本可用
- 如果失败，优先怀疑 barrier 顺序、shared 状态传播或 block 内阶段切换
- 它比 `03` 和 `05` 更强调“多次 barrier + 条件分支”的组合

## 备注

- 如果当前 `results/` 目录不存在，说明这个例子在本地还没被重新运行过
- README 中写的是**预期结果**，不是当前仓库快照里一定已经存在的结果文件
