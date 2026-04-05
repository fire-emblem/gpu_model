# 05 Softmax Reduction

## 例子作用

这个例子把 shared memory、barrier、归约和基础数值计算放在一起，是从“功能验证”过渡到“较复杂 kernel 验证”的关键样例。

它主要验证：

- block 内两阶段归约是否正确
- shared memory 与 barrier 组合是否稳定
- softmax 风格数值路径在三种执行模式下是否一致

## 本例子的关注点

- 第一阶段 `max` 归约是否正确
- 第二阶段 `sum(exp(x - max))` 归约是否正确
- 多次 `__syncthreads()` 是否把阶段边界正确隔开
- 最终 softmax 输出是否接近理论值

## 程序行为

这个 kernel 处理一个长度为 64 的 block 内向量：

1. 先把输入写入 shared memory
2. 做 block 内最大值归约
3. 计算 `exp(x - max)`
4. 再做一次 block 内求和归约
5. 输出 `e / sum`

当前 host 侧输入全为 1，因此理论输出应为：

- 每个位置都接近 `1 / 64`

这个设置的好处是预期结果非常清楚，便于定位到底是同步错误还是数值错误。

## 运行方式

```bash
./examples/05-softmax-reduction/run.sh
```

脚本会：

1. 编译 [softmax_reduction.hip](/data/gpu_model/examples/05-softmax-reduction/softmax_reduction.hip)
2. 分别以 `st` / `mt` / `cycle` 运行
3. 检查 `stdout.txt` 中是否包含 `softmax_reduction mismatches=0`

## 关键产物

运行后应生成：

- `results/softmax_reduction.out`
- `results/stdout.txt`
- `results/st`
- `results/mt`
- `results/cycle`

每个模式目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

如果当前脚本已经走到新的导出路径，重新运行后还可能看到：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准：

- 三个模式的 `stdout.txt` 都包含 `softmax_reduction mismatches=0`
- 三个模式的 `launch_summary.txt` 都应有 `ok=1`
- `trace.txt` 中应出现多次 shared memory 访问和同步事件
- host 比较应表明所有输出都接近 `1 / 64`

## 建议观察项

这个例子建议重点观察：

- 两次归约阶段之间是否被 barrier 正确分隔
- `cycle` 模式下不同阶段的执行密度是否变化明显
- `trace.txt` 中 shared 访问与 barrier 顺序是否合理

它比 `03` 更适合看“多阶段同步”，但仍然局限在单个 block 内。

## 调试入口

出现问题时建议按这个顺序排查：

1. `results/<mode>/stdout.txt`
   先看是否 `mismatches=0`
2. `results/<mode>/trace.txt`
   确认最大值归约和求和归约阶段是否完整
3. `results/<mode>/launch_summary.txt`
   看 kernel 是否完整结束
4. `timeline.perfetto.*`
   需要看分阶段时间线时再打开

## 结果解读

- 这个例子通过，说明“shared + barrier + 多阶段归约”基础路径大致正确
- 如果失败，不一定是数值函数本身错误，更常见的是 barrier 或 shared 阶段切换有问题
- 它是后面 reduction 类例子的中间基线

## 备注

- 仓库中的已有 `results/` 主要作为快照参考
- 默认重新运行 `run.sh` 会把结果写到 `.cache/example-results/05-softmax-reduction/`
- 若需要刷新仓库内快照，显式设置 `GPU_MODEL_EXAMPLE_RESULTS_MODE=repo`
- 如果重跑后出现 `.pb`，优先用它做 Perfetto 层级观察
