# 07 VecAdd Cycle Splitting

## 例子作用

这个例子不是为了验证“能不能做 vecadd”，而是为了比较**不同 HIP 写法对 cycle 开销的影响**。

它提供三个功能等价、实现方式不同的 vecadd 变体，适合回答这类问题：

- 为什么不同写法的 cycle 总数会不同
- trace / timeline 能否解释这种差异
- `st` / `mt` / `cycle` 的结果是否都保持正确

## 本例子的关注点

- 三种 vecadd 写法的 `cycle total_cycles` 是否不同
- 结果正确性是否在不同写法下都保持一致
- `cycle_comparison.txt` 是否能直接给出对比结论
- Perfetto/trace 是否能辅助解释开销差异

## 程序列表

- `vecadd_direct.hip`
  一元素一线程，最直接的 baseline
- `vecadd_grid_stride.hip`
  使用 grid-stride 循环，重点看额外控制流和重复访存
- `vecadd_chunk2.hip`
  单线程处理两个元素，重点看线程粒度变化对调度和访存的影响

## 运行方式

```bash
./examples/07-vecadd-cycle-splitting/run.sh
```

脚本会：

1. 分别编译三个 `.hip` 程序
2. 每个程序分别跑 `st` / `mt` / `cycle`
3. 汇总 `cycle` 模式下三个程序的 `total_cycles`
4. 写出 `results/cycle_comparison.txt`

## 关键产物

运行后应看到：

- `results/vecadd_direct.out`
- `results/vecadd_grid_stride.out`
- `results/vecadd_chunk2.out`
- `results/stdout.txt`
- `results/cycle_comparison.txt`
- `results/st/<variant>`
- `results/mt/<variant>`
- `results/cycle/<variant>`

每个模式子目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

重新按当前主线运行后，还应看到：


## 预期结果

预期成功标准：

- 三个变体在 `st` / `mt` / `cycle` 下都要通过 host 校验
- 每个模式目录的 `stdout.txt` 都应包含 `<variant> validation ok`
- `launch_summary.txt` 中 `ok=1`
- `results/cycle_comparison.txt` 中三种写法的 `total_cycles` 都是正整数
- 三种写法的 cycle 总数**不应全部完全相同**

这是本例最核心的预期之一。如果三个变体的 cycle 完全一样，这个对比就失去意义。

## 建议观察项

建议按这个顺序看：

1. `results/cycle_comparison.txt`
   先看三种写法的周期差异
2. 选一个 baseline
   例如 `vecadd_direct`
3. 再对照另一个变体
   例如 `vecadd_grid_stride` 或 `vecadd_chunk2`
4. 打开对应的：
   - `trace.txt`
   - `timeline.perfetto.json`

重点看：

- grid-stride 是否引入更多控制流
- chunk2 是否改变访存/计算分布
- resident/logical slot 语义是否与当前主线一致

## 调试入口

建议按这个顺序排查：

1. `results/stdout.txt`
   看整体脚本是否完成
2. `results/cycle_comparison.txt`
   看对比结果是否合理
3. `results/<mode>/<variant>/stdout.txt`
   看单个变体是否校验通过
4. `results/<mode>/<variant>/trace.txt`
   看具体指令与阶段分布
   需要看层级和时间线时再打开

## 结果解读

- 这个例子通过，说明“不同实现写法带来不同 cycle 成本”这条分析链路是可用的
- 它不追求复杂同步语义，而是追求“同功能不同写法”的结构性对比
- 这是做性能建模前很有价值的对照样例

