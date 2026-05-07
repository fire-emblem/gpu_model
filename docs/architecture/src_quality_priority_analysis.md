# `src` 质量问题优先级分析

> ⚠️ **分析快照**：本文档为某次质量扫描的分析记录，部分路径引用（如 `src/loader/`）对应旧目录结构，已不完全匹配当前代码。

本文基于当前 `src-only` 质量扫描结果，总结项目源码层面的关键问题、优先级排序和建议处理顺序。

扫描口径：

- 只扫描 `src/`
- `jscpd`：重复率
- `lizard`：圈复杂度 / 函数长度 / 参数数量
- `cppcheck`：静态检查

当前基线：

- `jscpd`
  - `sources=72`
  - `duplication=1.64%`
  - `clones=13`
- `lizard`
  - `functions=1466`
  - `warnings=68`
- `cppcheck`
  - `error=7`
  - `warning=0`
  - `style=60`
  - `performance=225`

补充约束：

- 本文优先级按“修复价值 / 风险 / 对后续重构的阻塞程度”排序
- 不按告警数量机械排序
- `passedByValue` 这类大量轻量性能/风格提示，不自动高于真实生命周期问题

## Priority 0: 严格门禁阻塞与真实缺陷嫌疑

这一层问题应最先处理。

原因：

- 会直接导致 `GPU_MODEL_QUALITY_STRICT=1` 失败
- 其中一部分明显带有生命周期/悬垂对象风险
- 如果不先清掉，后续引入 gate 或局部重构时噪音过大

### P0.1 生命周期 / 悬垂对象问题

这是最像真实 bug 的一组问题。

关键位置：

- [asm_parser.cpp](../../src/loader/asm_parser.cpp#L45)
  - `returnDanglingLifetime`
- [asm_parser.cpp](../../src/loader/asm_parser.cpp#L98)
  - `danglingTemporaryLifetime`
- [gcn_text_parser.cpp](../../src/loader/gcn_text_parser.cpp#L100)
  - `returnDanglingLifetime`

风险判断：

- 这些问题不只是风格告警，而是对象生存期与返回值语义可能不成立
- 位置集中在 loader / parser 主线上，容易在输入解析边界出错时暴露

建议动作：

1. 先核对返回类型是否持有了临时对象、局部 buffer 或 view
2. 如果返回的是 `string_view` / `span` / 引用，优先确认底层 owner 生命周期
3. 必要时改为显式拥有型返回值

### P0.2 `cppcheck internalAstError`

这组问题不一定都代表运行时 bug，但都代表代码表达已经对静态分析器不稳定。

关键位置：

- [encoded_gcn_encoding_def.cpp](../../src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp#L358)
  - `internalAstError`
- [device_image_loader.cpp](../../src/loader/device_image_loader.cpp#L43)
  - `internalAstError`
- [cycle_exec_engine.cpp](../../src/execution/cycle_exec_engine.cpp#L1110)
  - `internalAstError`
- [actual_timeline_builder.cpp](../../src/debug/timeline/actual_timeline_builder.cpp#L74)
  - `internalAstError`

风险判断：

- 这些点会阻塞严格门禁
- 同时说明相关表达式/初始化器/组合写法已经偏复杂或边界不清

建议动作：

1. 优先把复杂初始化表达式拆成若干中间变量
2. 减少嵌套构造、链式临时对象和过长的聚合初始化
3. 修复后重新跑 `GPU_MODEL_QUALITY_STRICT=1 bash scripts/run_quality_checks.sh`

## Priority 1: 核心执行路径复杂度过高

这层问题决定了后续架构改动的成本和回归风险。

最重热点如下：

- [cycle_exec_engine.cpp](../../src/execution/cycle_exec_engine.cpp#L1084)
  - `CycleExecEngine::Run`
  - `CCN=163`
  - `LEN=898`
- [asm_parser.cpp](../../src/loader/asm_parser.cpp#L152)
  - `AsmParser::Parse`
  - `CCN=123`
  - `LEN=378`
- [exec_engine.cpp](../../src/runtime/exec_engine.cpp#L184)
  - `ExecEngineImpl::Launch`
  - `CCN=89`
  - `LEN=341`
- [functional_exec_engine.cpp](../../src/execution/functional_exec_engine.cpp#L1390)
  - `FunctionalExecutionCoreImpl::ExecuteWave`
  - `CCN=55`
  - `LEN=330`

按文件聚合后，复杂度热点最明显的文件是：

- `src/execution/program_object_exec_engine.cpp`
  - `warns=24`
  - `max_ccn=33`
  - `max_len=205`
- `src/instruction/encoded/internal/encoded_gcn_encoding_def.cpp`
  - `warns=12`
  - `max_ccn=33`
- `src/execution/cycle_exec_engine.cpp`
  - `warns=10`
  - `max_ccn=163`
  - `max_len=898`
- `src/execution/encoded_semantic_handler.cpp`
  - `warns=8`
  - `max_ccn=62`
- `src/execution/internal/semantic_handler.cpp`
  - `warns=8`
  - `max_ccn=51`

风险判断：

- 这些函数本身就是项目主架构热点
- 在 `cycle` / `functional` / `encoded` 三条执行线都存在巨函数
- 继续在这些函数上堆逻辑，会使后续 issue policy、timeline、runtime 行为修复越来越难

建议动作：

1. 先拆主流程函数，不先追求“所有告警归零”
2. 优先把函数拆成阶段边界明确的子步骤
3. 对 `CycleExecEngine::Run` 优先按 tick 阶段拆
4. 对 `ExecEngineImpl::Launch` 优先按 launch 组织、runtime 选择、trace/统计收口拆

## Priority 2: Loader / Object ingestion 重复逻辑

重复率总量不高，但重复集中在同一主题链路上。

最明显的一组是 loader / object 读取入口：

- [object_reader.cpp](../../src/program/object_reader.cpp#L11)
- [asm_parser.cpp](../../src/loader/asm_parser.cpp#L15)
- [gcn_text_parser.cpp](../../src/loader/gcn_text_parser.cpp#L7)
- [encoded_program_object.cpp](../../src/program/encoded_program_object.cpp#L23)

`jscpd` 文件命中次数也说明了这一点：

- `src/program/object_reader.cpp`
  - `4` 次 clone 命中
- `src/loader/executable_image_io.cpp`
  - `3` 次 clone 命中

风险判断：

- 这组重复多半来自字符串修整、头部解析、I/O 序列化/反序列化与 object ingestion 公共步骤
- 如果后续继续扩 loader 能力，重复会继续扩散成行为分叉

建议动作：

1. 抽稳定的文本处理 / binary section 读写 helper
2. 统一 object read / parse / image load 的公共前处理
3. 避免把 parser 自己变成新的大杂烩 util 容器

## Priority 3: Debug / Trace 导出链路重复

当前 trace 相关重复主要集中在事件导出、timeline 视图和 recorder 组装。

明显重复点：

- [trace_event_export.cpp](../../src/debug/trace/trace_event_export.cpp#L121)
- [trace_event_export.cpp](../../src/debug/trace/trace_event_export.cpp#L151)
- [cycle_timeline.cpp](../../src/debug/timeline/cycle_timeline.cpp#L64)
- [cycle_timeline.cpp](../../src/debug/timeline/cycle_timeline.cpp#L86)
- [recorder.cpp](../../src/debug/recorder/recorder.cpp#L179)
- [recorder.cpp](../../src/debug/recorder/recorder.cpp#L200)

`jscpd` 命中也支持这一判断：

- `src/debug/recorder/recorder.cpp`
  - `3` 次 clone 命中
- `src/debug/trace/trace_event_export.cpp`
  - `2` 次 clone 命中
- `src/debug/timeline/cycle_timeline.cpp`
  - `2` 次 clone 命中

风险判断：

- trace 层按项目约束不能参与业务逻辑
- 如果导出/视图组装代码重复演化，后续最容易发生“表现层字段漂移”

建议动作：

1. 统一 recorder entry -> export view 的构造过程
2. 减少多处重复的字段拷贝 / 字段拼接逻辑
3. 保持“业务事件先发生，trace 只消费”的现有边界

## Priority 4: 大量低信号 `cppcheck` 建议

数量最多的是：

- `passedByValue=225`
- `useStlAlgorithm=38`
- `unassignedVariable=10`

按文件分布，`passedByValue` 噪音主要集中在：

- [instruction_builder.cpp](../../src/isa/instruction_builder.cpp)
  - `171` 条命中
- [program_object_exec_engine.cpp](../../src/execution/program_object_exec_engine.cpp)
  - `14` 条命中

风险判断：

- 这类问题短期内不会比 P0/P1 更关键
- 但它们会放大报告噪音，影响后续把扫描结果当日常门禁

建议动作：

1. 不要把这类问题排在生命周期和复杂度热点前面
2. 只在顺手重构相关文件时批量清理
3. 对 `std::string_view` 之类轻量类型保持判断，不要机械改签名

## 建议处理顺序

建议分三批推进：

### Batch A: 先恢复严格门禁可解释性

处理内容：

- 生命周期 / 悬垂对象问题
- 4 个 `internalAstError`

完成标志：

- `cppcheck error` 明显下降
- `GPU_MODEL_QUALITY_STRICT=1` 的失败点只剩可接受存量，或者直接通过

### Batch B: 再拆核心巨函数

处理内容：

- `CycleExecEngine::Run`
- `ExecEngineImpl::Launch`
- `AsmParser::Parse`
- `FunctionalExecutionCoreImpl::ExecuteWave`

完成标志：

- 主热点函数长度和复杂度明显下降
- 对应行为保持不变，回归测试通过

### Batch C: 最后处理重复与低信号告警

处理内容：

- loader / object ingestion 公共逻辑
- trace 导出公共逻辑
- `passedByValue` / `useStlAlgorithm` / `constVariable`

完成标志：

- `jscpd` clone 进一步下降
- `cppcheck` 摘要噪音减少

## 结论

当前最关键的问题，不是总重复率，也不是 `passedByValue` 数量，而是：

1. loader 里的生命周期问题
2. 让严格模式失败的 `internalAstError`
3. 执行主路径上的超大函数与高复杂度

换句话说：

- 先修“可能出错”和“阻塞门禁”的问题
- 再拆“架构会继续恶化”的热点函数
- 最后再做批量风格化清理
