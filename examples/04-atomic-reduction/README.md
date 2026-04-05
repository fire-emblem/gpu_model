# 04 Atomic Reduction

## 例子作用

这个例子用最小的 global atomic 归约验证“多线程并发写同一地址”这条路径。

它的目标不是性能，而是语义正确性：

- atomic 是否真的按原子读改写执行
- 多线程/多 wave 更新同一地址时结果是否稳定
- 三种执行模式下是否都得到同一个归约值

## 本例子的关注点

- global atomic 语义是否正确
- 并发更新同一地址时是否存在丢更新
- `st` / `mt` / `cycle` 是否得到一致的最终计数
- trace 中能否看到 atomic 对应的 memory 操作序列

## 程序行为

kernel 的逻辑非常简单：

- 若线程索引 `i < n`
- 就对单个输出地址执行一次 `atomicAdd(out, 1)`

host 最终检查：

- 输出值是否严格等于 `n`

因为这里所有线程都更新同一个地址，所以非常适合做原子语义验证。

## 运行方式

```bash
./examples/04-atomic-reduction/run.sh
```

脚本会：

1. 编译 [atomic_reduction.hip](/data/gpu_model/examples/04-atomic-reduction/atomic_reduction.hip)
2. 以 `st` / `mt` / `cycle` 运行
3. 检查 `stdout.txt` 中是否包含 `atomic_reduction value=257 expected=257`

## 关键产物

运行完成后应看到：

- `results/atomic_reduction.out`
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

按新主线重跑后，也可能看到：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准：

- `results/st/stdout.txt` 包含 `atomic_reduction value=257 expected=257`
- `results/mt/stdout.txt` 包含 `atomic_reduction value=257 expected=257`
- `results/cycle/stdout.txt` 包含 `atomic_reduction value=257 expected=257`
- 三个模式的 `launch_summary.txt` 中都应有 `ok=1`
- trace 中应能看到 atomic 对应的 memory 操作

## 建议观察项

这个例子建议重点看：

- 最终计数值是否严格等于线程数上限 `n`
- trace 中 atomic 操作的提交顺序
- `cycle` 模式下是否出现明显的内存串行化痕迹

如果结果不等于 `257`，这个例子就已经说明原子语义有问题。

## 调试入口

优先排查顺序：

1. `results/<mode>/stdout.txt`
   先看最终计数值
2. `results/<mode>/trace.txt`
   看 atomic 指令和 memory 相关事件
3. `results/<mode>/launch_summary.txt`
   确认 kernel 是否正常结束
4. `timeline.perfetto.*`
   用于观察 atomic 对时间线的影响

## 结果解读

- 这个例子通过，说明最基础的 global atomic 语义基本正确
- 它不覆盖 shared atomic，也不覆盖更复杂的竞争模式
- 但它是所有 atomic 相关路径的最小基线

## 备注

- 当前仓库中的 `results/` 主要作为快照参考
- 默认重新运行 `run.sh` 会把结果写到 `.cache/example-results/04-atomic-reduction/`
- 若需要刷新仓库内快照，显式设置 `GPU_MODEL_EXAMPLE_RESULTS_MODE=repo`
- 如果旧结果里还没有 `.pb`，属于结果目录未重跑，不代表功能缺失
