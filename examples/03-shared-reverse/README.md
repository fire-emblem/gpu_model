# 03 Shared Reverse

## 例子作用

这是第一个明确依赖 shared memory 和 block 内同步的真实 HIP 例子。

它用一个非常直观的“block 内逆序写回”来验证：

- shared memory 读写是否正确
- `__syncthreads()` 是否真的起到了同步作用
- block 内线程之间的数据可见性是否符合预期

## 本例子的关注点

- shared memory 搬运与读取是否正确
- barrier 前后数据是否一致
- block 内逆序写回结果是否完全匹配预期
- trace 中是否能看到 shared 访问与同步事件

## 程序行为

每个 block 会做两步：

1. 把输入元素写入 `__shared__` 数组
2. barrier 之后按反向索引读回并写到输出

host 会构造每个 block 的期望逆序结果，然后逐元素比较。

这是一个非常适合验证 block 内同步正确性的样例，因为预期结果直观、容易定位。

## 运行方式

```bash
./examples/03-shared-reverse/run.sh
```

脚本会：

1. 编译 [shared_reverse.hip](/data/gpu_model/examples/03-shared-reverse/shared_reverse.hip)
2. 以 `st` / `mt` / `cycle` 运行
3. 检查 `stdout.txt` 中是否包含 `shared_reverse mismatches=0`

## 关键产物

运行后应有：

- `results/shared_reverse.out`
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

重跑到新主线后，也可能看到：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准：

- 三个模式的 `stdout.txt` 都包含 `shared_reverse mismatches=0`
- 三个模式的 `launch_summary.txt` 都包含 `ok=1`
- `trace.txt` 中应能看到 shared memory 访问
- 同步相关 trace 不应缺失，且不应出现因为 barrier 问题导致的卡死

## 建议观察项

这个例子最值得观察的是：

- barrier 前后的 shared 访问顺序
- 一个 block 内线程的读写是否在 barrier 后才发生逆序读取
- `cycle` 模式是否能把 shared 访问和同步阶段分开

如果要看更复杂的多次 barrier 关系，应该转到 `08-conditional-multibarrier`。

## 调试入口

建议按以下顺序排查：

1. `results/<mode>/stdout.txt`
   看 `mismatches` 是否为 0
2. `results/<mode>/trace.txt`
   看 barrier 前后 shared memory 访问顺序
3. `results/<mode>/launch_summary.txt`
   确认是否正常结束
4. `timeline.perfetto.*`
   用于观察 shared / barrier 相关切片是否连贯

## 结果解读

- 如果这个例子通过，说明 block 内单次 barrier + shared memory 基础路径大致正确
- 如果失败，优先怀疑同步或 shared 地址读写顺序，而不是普通 ALU 逻辑
- 这是后面所有 shared/barrier 例子的前置基线

## 备注

- 仓库中的旧 `results/` 主要作为快照参考
- 默认重新运行 `run.sh` 会把结果写到 `.cache/example-results/03-shared-reverse/`
- 若需要刷新仓库内快照，显式设置 `GPU_MODEL_EXAMPLE_RESULTS_MODE=repo`
- 若结果目录里没有 `.pb`，请以重跑后的结果为准
