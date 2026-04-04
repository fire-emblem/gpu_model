# 09 Dynamic Shared Sum

## 例子作用

这个例子专门验证 `extern __shared__` 这条动态 shared memory 路径。

相比固定大小 shared memory，这里额外关注：

- shared 大小是否在 launch 时正确传入
- `extern __shared__` 地址是否正确
- block 内同步之后能否正确读回动态 shared 内容

## 本例子的关注点

- 动态 shared memory 大小是否正确传递
- `extern __shared__` 数组访问是否正确
- barrier 后 block 内求和是否正确
- 不同 block 的结果是否一致

## 程序行为

kernel 会在每个 block 内：

1. 把 `tid + 1` 写入动态 shared memory
2. barrier
3. 由 `tid == 0` 遍历整个 shared memory 做求和
4. 把该 block 的求和结果写到输出数组

当前参数下，每个 block 的理论结果是：

- `1 + 2 + ... + 64 = 2080`

因此 host 会检查所有 block 输出是否都等于这个值。

## 运行方式

```bash
./examples/09-dynamic-shared-sum/run.sh
```

脚本会：

1. 编译 [dynamic_shared_sum.hip](/data/gpu_model/examples/09-dynamic-shared-sum/dynamic_shared_sum.hip)
2. 分别以 `st` / `mt` / `cycle` 运行
3. 检查 `stdout.txt` 中是否包含 `dynamic_shared_sum mismatches=0`

## 关键产物

运行后应生成：

- `results/dynamic_shared_sum.out`
- `results/st`
- `results/mt`
- `results/cycle`

每个模式目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

重跑到当前主线后，还可能看到：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准：

- 三个模式的 `stdout.txt` 都包含 `dynamic_shared_sum mismatches=0`
- 三个模式的 `launch_summary.txt` 中都应有 `ok=1`
- host 侧每个 block 的输出都应等于 `2080`
- trace 中应能看到 shared memory 访问与 barrier

## 建议观察项

这个例子最适合看：

- 动态 shared 大小传递是否真的生效
- `cycle` 模式下 shared 访问和 barrier 是否按预期分段
- block 内由单个线程做最终遍历时的时序是否合理

它是检查动态 shared 路径是否接通的首选样例。

## 调试入口

建议按顺序排查：

1. `results/<mode>/stdout.txt`
   看 `mismatches` 是否为 0
2. `results/<mode>/trace.txt`
   看 shared 写入、barrier、最终求和阶段
3. `results/<mode>/launch_summary.txt`
   看运行是否完整结束
4. `timeline.perfetto.*`
   需要从时间线理解 shared 阶段时再打开

## 结果解读

- 这个例子通过，说明动态 shared memory 的最小可用路径已经接通
- 如果失败，优先怀疑 shared 大小传递、地址计算或 barrier 顺序
- 它和 `03` 的区别在于：`03` 验证固定 shared，`09` 验证动态 shared

## 备注

- 当前仓库中的 `results/` 可能是旧产物
- 若旧结果里没有 `.pb`，请以重跑后的结果为准
