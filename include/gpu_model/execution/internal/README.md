# `execution/internal`

这里存放 `execution/*` 的内部执行支撑件。

约束：

- 该目录是长期内部层，不是历史残留兼容层
- 对外公开接口应优先通过 `execution/*` 暴露
- 这里主要承载：
  - issue model / issue scheduler / eligibility
  - scoreboard / event queue
  - internal semantics / semantic handlers / op plan
  - tensor helper 等执行期内部工具

判断标准：

- 如果一个组件主要服务于执行引擎内部协作，而不适合作为稳定上层 API，就应留在这里
