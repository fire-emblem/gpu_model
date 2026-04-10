# 06 MMA GEMM

## 例子作用

这个例子不是常规功能样例，而是一个“能力探针”。

它的目的很明确：

- 检查当前环境是否支持 `gfx90a` / `mfma`
- 如果支持，验证最小 MMA/MFMA 路径是否能跑通
- 如果不支持，脚本是否能明确给出“当前环境不支持”的结果，而不是静默失败

## 本例子的关注点

- `hipcc --offload-arch=gfx90a` 是否可用
- `__builtin_amdgcn_mfma_f32_16x16x4f32` 对应路径是否能编译和运行
- 默认 `mt` 模式在“支持”场景下是否能给出正确结果
- 在“不支持”场景下是否输出清晰状态

## 程序行为

程序会先尝试编译 [mma_gemm.hip](mma_gemm.hip)。

分两种情况：

1. 当前环境支持 `gfx90a/mfma`

- 编译成功
- 默认运行 `mt`
- kernel 最终应写出 `4.0`

2. 当前环境不支持 `gfx90a/mfma`

- 编译失败
- 脚本不会继续硬跑
- 会在 `results/mt/stdout.txt` 中写入 `STATUS: unsupported_yet`

## 运行方式

```bash
./examples/06-mma-gemm/run.sh
```

## 关键产物

若编译成功，通常会有：

- `results/mma_gemm.out`
- `results/mt`

`results/mt` 目录下应有：

- `stdout.txt`
- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- `launch_summary.txt`

若编译失败：

- 一般只有 `results/mt/stdout.txt`

并明确写出 `unsupported_yet`

## 预期结果

这个例子的“预期结果”有两种，而且两种都可能是对的。

### 情况 A：环境支持 `gfx90a/mfma`

- `mt/stdout.txt` 应包含 `mma_gemm out=4.000000 expected=4.000000`
- `mt/launch_summary.txt` 中应有 `ok=1`
- trace 不应为空

### 情况 B：环境不支持 `gfx90a/mfma`

- `mt/stdout.txt` 应包含 `STATUS: unsupported_yet`
- 这属于预期行为，不算失败

## 建议观察项

这个例子最重要的不是 trace，而是先判断环境能力：

- 先看是否成功生成 `mma_gemm.out`
- 再看 `stdout.txt` 是“真正运行成功”还是“明确标记不支持”

只有在环境支持并且运行成功的前提下，才有必要继续看 trace 和 timeline。

## 调试入口

建议按这个顺序排查：

1. 先看 `results/mt/stdout.txt`
   判断是“不支持”还是“支持但运行失败”
2. 如果是不支持
   先处理工具链 / 目标架构问题，不要先看 trace
3. 如果是支持但运行失败
   再看：
   - `launch_summary.txt`
   - `trace.txt`
   - `timeline.perfetto.json`

## 结果解读

- 这个例子通过，不代表通用 GEMM 已经成熟，只代表最小 MMA/MFMA 探针接通
- 这个例子输出 `unsupported_yet`，也不代表仓库错误，只代表宿主环境暂时不支持该能力
- 它更像能力边界检测，而不是稳定功能基线

## 备注

- 这个例子天然依赖宿主工具链和目标架构支持，比前几个例子更不稳定
- README 中的“预期结果”必须接受两种分支，而不是只接受成功运行一种情况
