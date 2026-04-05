# 01 VecAdd Basic

## 例子作用

这是最小的端到端入门例子，用来确认整条路径已经接通：

- 真实 HIP 源码可以被 `hipcc` 编译成 `.out`
- `LD_PRELOAD` 方式的运行时拦截有效
- `st` / `mt` / `cycle` 三种执行模式都能跑通
- trace / timeline 能正常落盘

如果这个例子都不能稳定通过，后面的例子基本不值得继续看。

## 本例子的关注点

- 最基础的逐元素向量加法是否结果正确
- 三种执行模式是否给出一致的 host 校验结果
- 最基础的 trace、timeline、launch summary 是否都成功生成
- 是否能把它当作最小回归样例

## 程序行为

kernel 做的事情非常直接：

- 读取 `a[i]` 和 `b[i]`
- 计算 `c[i] = a[i] + b[i]`
- host 侧把结果拷回并逐元素校验

这里没有 shared memory、barrier、atomic 或复杂控制流，重点就是确认基础运行链路。

## 运行方式

```bash
./examples/01-vecadd-basic/run.sh
```

脚本会做三件事：

1. 用 `hipcc` 编译 [vecadd.hip](/data/gpu_model/examples/01-vecadd-basic/vecadd.hip)
2. 通过 `gpu_model_hip_interposer` 分别运行 `st` / `mt` / `cycle`
3. 检查每个模式的 `stdout.txt` 中是否包含 `vecadd validation ok`

## 关键产物

运行完成后，结果目录通常是：

- `results/vecadd.out`
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

如果当前运行脚本已经接入了新的 native Perfetto 导出，重新运行后还应看到：

- `timeline.perfetto.pb`

## 预期结果

预期成功标准非常明确：

- `results/st/stdout.txt` 包含 `vecadd validation ok`
- `results/mt/stdout.txt` 包含 `vecadd validation ok`
- `results/cycle/stdout.txt` 包含 `vecadd validation ok`
- 三个模式的 `launch_summary.txt` 中都应有 `ok=1`
- 三个模式的 `trace.txt` / `trace.jsonl` 都不应为空
- trace 中应能看到至少这些事件：
  - `Launch`
  - `WaveStep`
  - `Commit`
  - `WaveExit`

如果重新生成了 `.pb`：

- `timeline.perfetto.pb` 应能在 Perfetto 中正常打开
- `timeline.perfetto.json` 应能作为文本检查版本正常解析

## 建议观察项

第一次看 examples 时，建议只看三样：

- `results/st/stdout.txt`
- `results/st/launch_summary.txt`
- `results/st/trace.txt`

这个例子不适合看复杂调度现象，但很适合确认：

- kernel 是否被正确识别
- launch 是否成功
- 最基础的波前执行和提交事件是否存在

## 调试入口

如果失败，按这个顺序排查：

1. 看 `results/<mode>/stdout.txt`
   先确认是编译失败、运行失败，还是结果校验失败
2. 看 `results/<mode>/launch_summary.txt`
   确认 `ok`、`begin_cycle`、`end_cycle`、`total_cycles`
3. 看 `results/<mode>/trace.txt`
   确认是否连 `Launch` / `WaveStep` 都没有产生
4. 再看 `timeline.perfetto.json` 或 `timeline.perfetto.pb`
   只在需要看时间线时使用

## 结果解读

- 这个例子通过，只能说明“最小链路通了”
- 它不能说明 barrier、shared memory、atomic、waitcnt 等复杂路径一定正确
- 但它是后续所有例子的前置基线

## 备注

- 仓库中现有 `results/` 主要作为快照参考
- 默认重新运行 `run.sh` 会把结果写到 `.cache/example-results/01-vecadd-basic/`
- 若需要刷新仓库内快照，显式设置 `GPU_MODEL_EXAMPLE_RESULTS_MODE=repo`
- 如果旧结果里只有 `timeline.perfetto.json` 而没有 `.pb`，不表示功能缺失，只表示结果目录还没按新逻辑重跑
