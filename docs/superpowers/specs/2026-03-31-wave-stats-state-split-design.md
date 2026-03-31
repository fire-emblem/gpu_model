# Wave Stats State Split Design

## Goal

扩展现有 `WaveStats` trace 快照，在 kernel 级总量快照中增加：

- `runnable`
- `waiting`

并明确与已有字段的关系：

- `active = runnable + waiting`
- `launch = active + end`

用于更直接地分析程序进度、占用以及“在跑”与“在等”的分布。

## Scope

本轮仍然只做 `FunctionalExecEngine` 主线的 trace 快照扩展。

本轮范围：

- 扩展现有 `WaveStats` message
- 保持 `TraceEventKind::WaveStats` 不变
- 保持 kernel 级总量快照，不拆 block / AP / PEU
- 更新 trace regressions 与 barrier/waitcnt 场景验证

本轮明确不做：

- block 级 `WaveStats`
- occupancy 百分比
- `CycleExecEngine` / `EncodedExecEngine` 跟进
- 每条 `WaveStep` 都发统计

## Current Problem

现有 `WaveStats` 已提供：

- `launch`
- `init`
- `active`
- `end`

这已经能看总进度，但还不能直接区分：

- 当前多少 wave 真正在运行
- 当前多少 wave 卡在 barrier / waitcnt 等等待态

如果只看 `active`，会把：

- 真正在推进的 runnable wave
- 仍处于生命周期内但正在 waiting 的 wave

混在一起，不利于分析程序“为什么不前进”。

## Design Summary

本轮不新增新的 trace event kind，而是扩展现有 `WaveStats` message。

推荐格式：

```text
launch=128 init=128 active=64 runnable=40 waiting=24 end=64
```

这里：

- `runnable`
  - 当前 `WaveRunState::Runnable`
- `waiting`
  - 当前 `WaveRunState::Waiting`
- `end`
  - 当前 `WaveRunState::Completed`
- `active`
  - 不再单独口径扫描定义，而是显式保持 `active = runnable + waiting`

## Counter Definitions

### launch

已发出 `WaveLaunch` 的 wave 总数。

### init

已完成 wave 初始状态建立的总数。

当前 functional 主线上，`init` 仍然等于已 materialize 的 wave 数，通常与 `launch` 相等。  
本轮保留它，是为了后续如果 init / launch 分开，不需要再改 schema。

### runnable

当前 `run_state == WaveRunState::Runnable` 的 wave 数。

### waiting

当前 `run_state == WaveRunState::Waiting` 的 wave 数。

### end

当前 `run_state == WaveRunState::Completed` 的 wave 数。

### active

定义为：

```text
active = runnable + waiting
```

不再独立推导“active but not completed”的另一套口径，以避免将来状态变更后漂移。

## Functional Executor Integration

这轮仍沿用上一批已确定的发射点：

1. `WaveLaunch` 批量发完后
2. barrier release 后
3. waitcnt resume 后
4. wave end 后
5. kernel 结束前

也就是说：

- 只在生命周期变化点发快照
- 不在每个 `WaveStep` 后都发

## Implementation Rule

统计值应直接从当前 `WaveContext` 的显式状态读取，而不是维护一套额外缓存：

- `Runnable` -> `runnable`
- `Waiting` -> `waiting`
- `Completed` -> `end`

这样做的好处：

- 不会出现“状态字段”和“统计缓存”双写不同步
- 后续如果新增更多 wait reason，统计天然自动跟上

## Trace Representation

继续沿用：

- `kind = TraceEventKind::WaveStats`
- `message = stable key=value string`

建议 key 顺序固定为：

```text
launch
init
active
runnable
waiting
end
```

固定顺序便于：

- 文本 trace 直接目视比较
- JSON trace message 做简单解析
- regression 直接断言字符串

## Testing Strategy

### 1. TraceTest 基础不变量

在 `TraceTest` 中增加基础断言，验证：

- 初始快照存在 `runnable`
- 初始快照通常为 `waiting=0`
- 最终快照为 `active=0 runnable=0 waiting=0 end=launch`
- 所有断言的快照都满足 `active = runnable + waiting`

### 2. Barrier Functional Trace Regression

在已有 shared-barrier regression 中验证：

- 中间 `WaveStats` 出现 `waiting>0`
- barrier release 后存在 `runnable` 回升的快照

### 3. Waitcnt Functional Trace Regression

在已有 waitcnt regression 中验证：

- 显式 `s_waitcnt` 导致的中间快照出现 `waiting>0`
- wait 满足后 `waiting` 降低、`runnable` 回升

## Acceptance Criteria

本轮完成标准：

1. `WaveStats` message 已扩展到 `runnable` / `waiting`
2. 对回归覆盖到的所有快照，都满足：
   - `active = runnable + waiting`
   - `launch = active + end`
3. 至少一组 barrier 场景看到 `waiting>0`
4. 至少一组 waitcnt 场景看到 `waiting>0`
5. `TraceTest.*` 与受影响 functional trace ring 通过

## Approaches Considered

### Option A: 扩展现有 WaveStats

优点：

- 复用当前事件类型
- 不引入第二套 progress event
- 便于对照前一批结果

缺点：

- message 更长一些

这是推荐方案。

### Option B: 新增第二个详细状态事件

优点：

- 语义更细

缺点：

- 和现有 `WaveStats` 重复
- trace 消费端更复杂

不推荐。

### Option C: 直接上 block 级分拆

优点：

- 可看到局部不均衡

缺点：

- trace 量显著增加
- 当前阶段收益不如先把 kernel 级状态拆细

不推荐作为当前顺序。

## Recommended Next Step

在本设计获批后，先写 implementation plan。第一任务应是先补 `TraceTest` 的初始/最终不变量回归，然后再补 barrier/waitcnt 中间态回归。
