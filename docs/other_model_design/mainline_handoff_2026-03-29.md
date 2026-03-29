# Mainline Handoff

日期：`2026-03-29`

适用背景：

- 主线已合入 `isa-fast` 批量 ISA 支持和 asm fixture 基础设施
- 主线已补入 tensor/AGPR/typed tensor ABI 第一批能力
- 已明确暂停继续铺更多 tensor shape，优先回到主线收口

---

## 1. 当前状态

当前主线已经具备：

- `encoded/raw_gcn` 路径与 `llvm-mc` 驱动的 `.s -> code object -> parser` 测试框架
- 一批 MFMA / tensor core 指令的 decode / loader / runtime / semantics 覆盖
- `agpr_count` / `accum_offset` 进入 loader、runtime API、launch trace
- `AGPRFile`、`v_accvgpr_read_b32`、`v_accvgpr_write_b32` 基础能力
- tensor op 在 timeline / trace 中的显式标注

这说明主线已经从“只有普通指令 functional/cycle”推进到“开始具备 tensor-aware runtime/ISA 骨架”。

---

## 2. 需要记录的当前问题

## 2.1 tensor 结果存储语义仍是过渡态

当前 `v_mfma_*` 语义为了兼容已有 runtime/测试，仍采取：

- 同时写 `VGPR`
- 同时写 `AGPR`

这不是长期稳态。

当前真正未决的问题是：

- tensor 结果的规范来源到底是 `AGPR-first`，还是继续保留 `VGPR mirror`
- 非 tensor 使用者是否会错误依赖当前的 `VGPR mirror`
- 哪一层负责兼容性别名，语义层还是 ABI/trace 层

结论：

- 后续应把 `AGPR` 作为 tensor 结果的 canonical storage
- 若需要兼容过渡，别名应尽量放在边界层，而不是核心语义层长期保留

## 2.2 encoded / modeled 分层方向已清楚，但实现还没完全收口

之前讨论已经明确：

- `raw / canonical` 命名长期不够清晰
- 更准确的长期术语应是 `encoded ISA` / `modeled ISA`
- `encoding descriptor`、`decoded instruction`、`executable instruction` 应继续拆层

当前问题不在于方向，而在于：

- 代码和测试中仍残留历史命名
- 一部分执行绑定逻辑仍和 decode 表示耦合
- shared effect / apply 层还没有完全抽出来

结论：

- 这是长期维护框架问题
- 可以分批推进，但不能再回到“继续叠局部特例”的路径

## 2.3 当前 ISA coverage 总数不是完整 LLVM/TD universe

之前统计里的 `79` / `82` 这一类数字，只代表：

- 当前项目“已纳入跟踪”的指令集合

并不代表：

- LLVM `AMDGPUAsmGFX9` 全量指令宇宙
- AMD TableGen/TD 全量 opcode 宇宙

因此当前覆盖率存在两个容易误解的点：

- 分母是 tracked subset，不是 full ISA universe
- 如果不分 encoding class / semantic family，数字会显得虚高

结论：

- coverage 报表必须明确区分：
  - tracked modeled/encoded coverage
  - full ISA universe coverage
  - unsupported but observed coverage
- 还需要加入不同 encoding 大类视图，避免“总数看起来满了，但只是局部集合满了”

## 2.4 主线最需要补的是共享执行层，不是继续零散扩 opcode

结合 `project_comparison_analysis_2026-03-29.md` 的结论，主线 P0 仍然是：

- `functional`
- `cycle`
- `raw/encoded`

三套执行路径的共享层收口。

当前真正的风险是：

- block/wave 状态构造重复
- memory helper 重复
- sync/barrier helper 重复
- effect apply / writeback 逻辑分散

这类问题会比“少一个 opcode”更快演化成维护灾难。

## 2.5 runtime / ABI 完整性仍是主路径缺口

当前 compute/tensor 支持比 runtime/ABI 收口进度更快。

仍需补齐的主线缺口包括：

- by-value aggregate / struct kernarg packing
- hidden arg / implicit arg 更完整覆盖
- relocation / symbol / bss / section address 关系
- module 生命周期
- runtime property / attribute / error behavior 完整性

结论：

- 如果这些边界不补，后续继续扩 ISA 收益会被 runtime 缺口吞掉

## 2.6 tensor 下一阶段应转向行为真实性，而不是 shape 数量

已经明确不再继续大量补 shape。

后续 tensor 优先级应转向：

- AGPR-first 结果模型
- tensor op end-to-end 行为覆盖
- 与 loader/runtime/trace 的一致性
- 更清晰的 tensor-specific regression gate

---

## 3. 回到主线后的建议优先级

## P0

1. 为 tensor 结果引入明确的 storage policy，逐步从 `VGPR+AGPR mirror` 过渡到 `AGPR-first`
2. 建立按 `encoding class + semantic family + observed unsupported` 的 coverage 看板
3. 按 `docs/plans/2026-03-29-exec-shared-epic-kickoff.md` 启动共享执行层收口
4. 建 runtime/ABI 缺口清单，并先补 by-value aggregate kernarg

## P1

1. 继续推进 `encoded` / `modeled` 长期命名与分层
2. 建立 tensor 行为回归测试，覆盖 `mfma -> accvgpr read/write -> trace/launch`
3. 补 module-load / metadata / ABI regression corpus

---

## 4. 一个关键判断

当前主线不缺“大方向”。

当前缺的是“把已经出现的正确方向做成长期稳定框架”。

如果继续把精力放在：

- 继续补更多 shape
- 继续零散堆 opcode
- 继续在未共享的执行层上叠特例

那么后续维护成本会快速上升。

所以回到主线后的正确动作应是：

- 收共享层
- 收 ABI/runtime 边界
- 收 coverage 分母定义
- 收 tensor 结果存储语义
