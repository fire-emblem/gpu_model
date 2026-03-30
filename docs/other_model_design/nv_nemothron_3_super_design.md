# 轻量级 AMDGPU Cycle 仿真器设计方案  

> [!NOTE]
> 外部参考/对比文档。用于记录其他方案与比较分析，不直接定义当前仓库实现。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。

*面向编译器与算子库的指令/算法优化分析*  

> **关键点**  
> - 仅在 **功能级别** 保证正确性（与原始 CPU 执行结果一致），不追求硬件 RTL 级周期精度。  
> - 以 **波（wave，64 线程）** 为最小调度单元的周期模型，能够统计 **整体程序运行周期**、 **波空泡（bubble）**、 **功能单元空闲** 等指标。  
> - 使用 **Perfetto** 生成时间线 trace，便于可视化分析。  
> - 通过 **LD_PRELOAD** hook 拦截 HIP/Runtime API（如 `hipLaunchKernelGGL`），零侵入式注入。  

---  

## 1. 设计目标与假设  

| 目标 | 说明 |
|------|------|
| 功能正确 | Host‑side CPU 执行保证最终输出与原始 `.out` 完全一致。 |
| 轻量周期模型 | 以 **wave** 为粒度，模拟指令发射、执行延迟、资源冲突产生的周期消耗。 |
| 可观测空泡 | 能够统计每个 wave 在每个周期内因资源不可用或同步等待而空闲（bubble）。 |
| 可视化 | 生成 Perfetto 兼容的 trace 文件（`trace.pb`），支持火焰图、时间线等分析。 |
| 易于插桩 | 通过动态库（`.so`）hook 拦截 HIP/Runtime API，零侵入。 |
| 非 RTL 级 | 不模拟流水线细节、寄存器重命名、指令合并等硬件细节；仅保留宏观延迟与冲突模型。 |

**假设**  

1. 输入为已编译好的 AMDGPU 二进制可执行文件（`.out`，ELF 格式），包含 `.text`、`.data` 以及 `amdgpu_metadata` 节（含 kernel 名、grid/block dim、私有/共享/常量内存大小等）。  
2. Host 端采用原生 X86/ARM CPU 执行，负责内存分配、数据拷贝、kernel 启动调度。  
3. 所有 GPU 指令在 **wave** 层面被解释为宏指令序列；每条指令有固定 **延迟**（latency）和 **资源占用**（VALU、SALU、LDS、TEX），这些参数来源于 AMD ISA 手册或简单查表（如 `S_MOV_B32` 延迟 1 cycle，占 VALU）。  
4. 波调度采用 **简化调度器**：每个周期检查所有就绪的 wave，若有可用的执行单元（CU）则发射；否则产生 bubble。  
5. 同步指令（`s_barrier`、`s_waitcnt`）导致波进入 **等待状态**，直至计数清零或所有波到达屏障。  ---  

## 2. 高层设计（High‑Level Design）  ### 2.1 系统组成  

```
+-------------------+        +-------------------+        +-------------------+
|   Host Program    |  --->  |   Runtime Hook    |  --->  |   Cycle Simulator |
|   (original .out) |        | (LD_PRELOAD/dll)  |        |   (libcycle.so)   |
+-------------------+        +-------------------+        +-------------------+
          |                         |                         |
          | 1. 启动时加载 hook       | 2. 拦截 hipLaunchKernelGGL,
          |    (LD_PRELOAD)          |    hipMemcpy 等 API    |
          v                         v                         v
+-------------------+        +-------------------+        +-------------------+
|   CPU Executer    | <---   |   Binary Parser   | <---   |   ELF Reader      |
|   (native code)   |        |   (sections,      |        |   (code/data/meta)|
|                   |        |    metadata)      |        +-------------------+
+-------------------+        +-------------------+                |
          |                         |                            |
          | 3. 执行 host 部分        | 4. 解码指令流 → InstrObj[]   |
          v                         v                            v
+-------------------+        +-------------------+        +-------------------+
|   Memory System   |        |   Wave Builder    |        |   Instr Decoder   |
|   (malloc/free)   |        |   (grid/block →   |        |   (ISA table)     |
|                   |        |    waves)         |        +-------------------+
+-------------------+        +-------------------+                |
          |                         |                            |
          | 5. 数据拷贝 (H2D/D2H)    | 6. 调度器 (WaveScheduler)   |
          v                         v                            v
+-------------------+        +-------------------+        +-------------------+
|   Perfetto Collector|    |   Execute Engine   |        |   Statistics      |
|   (trace events)    |    |   (issue, latency, |        |   (cycles, bubbles|
|                   |    |    resource use)    |        |    , stalls)      |
+-------------------+        +-------------------+        +-------------------+
```

### 2.2 数据流  1. **启动阶段**     - 通过 `LD_PRELOAD=libcycle.so` 注入动态库。     - `libcycle.so` 在 `dlopen` 时读取 `/proc/self/exe`，解析 ELF，提取 `.text`、`.data`、`amdgpu_metadata`。  

2. **Kernel 拦截**  
   - 对 `hipLaunchKernelGGL`、`hipMemcpy` 等函数做函数替换（使用 `dlsym(RTLD_NEXT, …)` 调用原始实现）。  
   - 在拦截包裹里：       a. 根据 metadata 创建 **WavePool**（每个 block 对应一组 wave）。  
     b. 将 kernel 参数拷贝到 **模拟内存**（Host 端分配的缓冲区，保持与真实 GPU 内存布局一致）。  
     c. 调用 `runKernel(kernelInfo, args)`。  
     d. 仿真结束后，如有需要将结果从模拟内存拷贝回 Host，再调用原始实现完成实际 GPU 执行（可选：仅用于验证正确性）。  

3. **仿真内核**  
   - **Wave Builder**：依据 `gridDim`、`blockDim`、`waveSize=64` 生成若干 `Wave` 对象；每个 Wave 持有：  
     - `pc`（指向待执行的 `InstrObj`）  
     - `activeMask`（64 位掩码，表示活跃线程）       - `waitCnt`（`s_waitcnt` 计数）  
     - `barrierState`（屏障阶段）  
   - **Instruction Decoder**：将 `.text` 中的原始机器码（4 字节或 8 字节）翻译成 `InstrObj`：  

     ```cpp
     struct InstrObj {
         OpCode        op;               // 枚举：S_MOV_B32, V_ADD_F32, S_BARRIER, …
         uint32_t      latency;          // 周期延迟（≥1）
         ResourceSet   res;              // 所需功能单元（VALU, SALU, LDS, TEX…）
         uint32_t      src[3];
         uint32_t      dst;
         bool          isSync;           // 是否同步指令
         std::string   disasm;           // 便于调试
     };
     ```  

   - **Wave Scheduler**（周期驱动）：  
     - 维护一个 **就绪队列**（所有 `state == READY` 的 wave）。  
     - 每个周期 `tick`：  
       1. 更新执行管道（完成延迟的指令回写）。  
       2. 处理完成的指令（写寄存器/内存、更新状态）。  
       3. 选取可发射的 wave：检查资源冲突；若可发射则发射当前波的下一条指令，更新 `pc`、`state`；否则记录 **bubble**。  
       4. 处理同步指令导致的状态变化（如 `s_barrier` 导致所有波进入 `BARRIER_WAIT`，`s_waitcnt` 计数递减至零时变为 `READY`）。  
       5. 检查是否所有 wave 均进入 `FINISHED` 状态 → 结束仿真。  

4. **统计与 Perfetto**  
   - 每个周期结束后记录：`totalCycles++`、`idleCycles += numberOfIdleCU`（可细分为 VALUIdle、SALUIdle、LDSIdle 等），`waveStalls[waveID]++`（若 wave 因同步/资源而未被调度）。  
   - 使用 **Perfetto C++ SDK** 在每个周期产生 `TraceEvent`：  
     - `name: "WaveExec"`，`waveID`、`instrIdx`、`latency`、`resource`  
     - `name: "Bubble"`，`cycle`、`idleVALU`、`idleSALU`、`idleLDS`、`idleTEX`  
   - 仿真结束后写入文件 `trace.pb`，可用 Perfetto UI 或 `traceconv` 转为 `json`/`html` 查看。  

### 2.3 关键接口  | 接口 | 说明 | 所在模块 |
|------|------|----------|
| `bool initSimulator(const std::string& elfPath);` | 加载 ELF，解析 metadata，构建指令表 | `ELFReader` / `BinaryParser` |
| `void launchKernel(const KernelInfo& info, void** args);` | 被 Runtime Hook 调用，创建 WavePool、拷贝参数、启动仿真 | `WaveBuilder` + `ExecuteEngine` |
| `void tick();` | 单周期调度驱动 | `Scheduler` |
| `std::vector<WaveStat> getStats() const;` | 返回每个 wave 的周期、丢失、空泡等统计 | `Statistics` |
| `void writePerfettoTrace(const std::string& outFile);` | 生成 trace.pb | `PerfettoCollector` |
| `bool verifyResult(const void* hostPtr, size_t size);` | 对比 Host 端 CPU 执行结果与模拟内存（可选） | `Validator` |  

---  

## 3. 低层设计（Low‑Level Design）  

### 3.1 模块划分  

| 模块 | 职责 | 核心类 / 函数 |
|------|------|---------------|
| **ELFReader** | 打开 ELF，读取段头、符号表、amdgpu_metadata | `ElfFile`, `parseMetadata()` |
| **BinaryParser** | 将 `.text` 按指令长度切片，交给解码器 | `InstrStream`, `nextChunk()` |
| **InstrDecoder** | 基于 ISA 表（硬编码或 JSON）将机器码转为 `InstrObj` | `decodeInstruction(uint32_t word)` |
| **WaveBuilder** | 根据 grid/block 生成 Wave 对象，分配私有/共享内存索引 | `buildWaves(dim3 grid, dim3 block)` |
| **Scheduler** | 周期级调度、资源冲突检测、波状态机 | `tick()`, `selectWave()` |
| **ExecuteEngine** | 维护执行管道、执行指令的副作用（寄存器/内存更新） | `issueInstr(Wave& w, InstrObj* i)` |
| **MemorySystem** | 模拟全局、共享、常量、私有内存（简单的字节数组 + 延迟模型） | `load(addr, size)`, `store(addr, size, data)` |
| **Statistics** | 累计周期、空泡、资源利用率、波直方图 | `recordCycle()`, `recordBubble()` |
| **PerfettoCollector** | 使用 Perfetto SDK 写 trace events | `emitWaveEvent()`, `emitBubbleEvent()` |
| **Validator** (可选) | 调用原始 CPU 版本 kernel（或已知正确答案）对比结果 | `compareBuffers()` |  

### 3.2 核心数据结构  

```cpp
// ---- InstrObj -------------------------------------------------
enum class OpCode : uint16_t { S_MOV_B32, V_ADD_F32, S_BARRIER, S_WAITCNT, … };
struct ResourceSet {
    bool valu : 1;
    bool salu : 1;
    bool lds  : 1;
    bool tex  : 1;
    // …可按需扩展
};
struct InstrObj {
    OpCode        op;
    uint32_t      latency;        // 周期延迟（≥1）
    ResourceSet   res;
    uint32_t      src[3];
    uint32_t      dst;
    bool          isSync;         // s_barrier / s_waitcnt 等
    std::string   disasm;         // 便于调试
};

// ---- Wave -----------------------------------------------------
enum class WaveState { READY, ISSUED, WAIT_SYNC, BARRIER_WAIT, FINISHED };
struct Wave {
    uint32_t      wid;            // 全局波 ID
    uint32_t      bid;            // 所属 block
    uint32_t      pc;             // 指向 InstrPool 的索引    WaveState     state;
    uint64_t      activeMask;     // 64-bit，哪些线程活跃    uint32_t      waitCnt;        // s_waitcnt 剩余计数    uint32_t      barrierGen;     // 屏障代数
    // 私有寄存器文件（简化为向量）
    std::array<uint32_t, 256>    vgpr;   // 视具体而定
    std::array<uint32_t, 16>     sgpr;
};

// ---- Statistics ------------------------------------------------
struct WaveStat {
    uint32_t wid;
    uint64_t cycles;          // 该波从 ISSUED 到 FINISHED 经过的周期数
    uint64_t stalls;          // 因同步/资源被延迟的周期数    uint64_t bubbles;         // 波本身在就绪但未被调度的周期数（可选）
    double   utilValu;        // VALU 利用率 = issued cycles / total cycles
    // …其他资源利用率
};
```  

### 3.3 关键算法  

#### 3.3.1 二进制解析与指令解码  

1. **读取 ELF**  
   - 使用 `libelf`（或手动解析 ELF64 头）定位 `.text` 段的文件偏移 `sh_offset` 和大小 `sh_size`。  
   - 将该段直接 `mmap` 为只读内存块，得到 `uint8_t* codeBase`。  2. **切片**  
   - AMDGPU ISA 定义：标量指令 4 字节，向量指令 4 字节（有时 8 字节的双字指令）。  
   - 遍历 `codeBase`，每次读取 4 字节；若高位位模式匹配到 **双字指令**（如 `V_MADMK_F32`），则再读取接下来的 4 字节组成 8 字节指令。  3. **解码表**  
   - 预先生成哈希表：`opcode → (latency, resourceSet, disasmTemplate)`。  
   - 解码函数伪码：       ```cpp
     InstrObj decodeInstr(const uint8_t* ptr, size_t bytes) {
         uint32_t word = le32(ptr);          // 小端序读取
         auto it = opcodeTable.find(word & OPCODE_MASK);
         if (it == end(opcodeTable)) throw UnsupportedInstr;
         InstrObj i;
         i.op      = it->second.opcode;
         i.latency = it->second.latency;
         i.res     = it->second.resources;
         // 根据格式字段提取 src/dst/imm等（参考 AMD ISA 手册）
         i.disasm  = formatDisasm(it->second.tmpl, word);
         i.isSync  = (i.op == OpCode::S_BARRIER || i.op == OpCode::S_WAITCNT);
         return i;
     }
     ```  #### 3.3.2 Wave 生成与调度  

- **波数计算**  

  $$
  \text{wavesPerBlock} = \left\lceil \frac{\text{blockDim.x} \times \text{blockDim.y} \times \text{blockDim.z}}{\text{waveSize}} \right\rceil
  $$

  `totalWaves = gridDim.x * gridDim.y * gridDim.z * wavesPerBlock`  

- **Wave 初始化**  
  - 为每个 wave 分配唯一 `wid`。  
  - `pc` 初始化为 kernel 的入口偏移（从 metadata 中获得）。  
  - `activeMask` 初始化为全 1（所有线程活跃），除非有显式的 `sgpr` 基于 `tid` 的条件分支。    - 私有寄存器文件按需分配（可采用懒分配）。  

- **调度循环**（简化版）  

  ```cpp
  void Scheduler::run() {
      while (!allFinished()) {
          tick();
          ++totalCycles_;
      }
  }

  void Scheduler::tick() {
      // 1. 更新执行管道（完成延迟的指令回写）
      executePipeline_.advance();   // 返回已完成的 (wave, instr) 对

      // 2. 处理完成的指令（写寄存器/内存、更新状态）
      for (auto& [w, instr] : executePipeline_.finished()) {
          executeInstr(w, instr);
      }

      // 3. 选取可发射的 wave
      Wave* cand = selectReadyWave();
      if (cand) {
          // 发射当前波的下一条指令          InstrObj* nextInstr = &instrPool[cand->pc];
          if (checkResources(cand, nextInstr)) {
              issueWave(cand, nextInstr);
              cand->state = WaveState::ISSUED;
              cand->pc++;   // 指向下一条
          } else {
              // 资源冲突 → 本周期产生 bubble
              stats_.recordBubble(cand->wid, executePipeline_.getIdleResources());
          }
      } else {
          // 没有就绪波 → 全部 bubble
          stats_.recordGlobalBubble(executePipeline_.getIdleResources());
      }
  }
  ```

  - `selectReadyWave()` 可实现为 **轮询**（公平）或 **优先级**（如倾向于已等待最长时间的 wave）。  

#### 3.3.3 指令执行与周期计数  

- **执行副作用**（简化版）  

  ```cpp
  void Scheduler::executeInstr(Wave& w, const InstrObj& i) {
      switch (i.op) {
          case OpCode::V_ADD_F32: {
              float a = uint2float(w.vgpr[i.src[0]]);
              float b = uint2float(w.vgpr[i.src[1]]);
              float r = a + b;
              w.vgpr[i.dst] = float2uint(r);
              break;
          }
          case OpCode::S_MOV_B32: {
              w.sgpr[i.dst] = i.src[0];   // 立即数或寄存器
              break;
          }
          case OpCode::S_BARRIER: {
              w.state = WaveState::BARRIER_WAIT;
              ++globalBarrierGen;
              break;
          }
          case OpCode::S_WAITCNT: {
              w.waitCnt = i.src[0];   // 设定等待计数
              w.state   = WaveState::WAIT_SYNC;
              break;
          }
          // …其它指令
      }
      // 访存指令调用 MemorySystem，若未命中则额外增加延迟（可选）
  }
  ```  

- **周期计数**  
  - 每成功 `issueWave` 计为 **1 个发射周期**。  
  - 指令的 **latency** 通过让其在执行管道中停留 `latency-1` 个周期实现（发射周期已计 1 次）。  
  - 总周期 `totalCycles_` 在主循环结束后得到。  

#### 3.3.4 同步与空泡检测  

| 同步类型 | 检测方式 | 产生的状态变化 |
|----------|----------|----------------|
| `s_barrier` | 所有属于同一 block 的 wave 进入 `BARRIER_WAIT` 后，计数器 `barrierArrived[blockID]++`；当达到 `wavesPerBlock` 时，全部波状态转为 `READY`，并清空计数器。 | 波在等待期间计为 **stall**（不消耗发射资源，但占用周期）。 |
| `s_waitcnt` | 波的 `waitCnt` 指示需要等待的 **计数器**（如 `lgkmcnt`、`vmcnt`）。模拟中，每发射一条 **内存指令**（VALU/TEX/LDS）会递减对应全局计数器；当计数归零时，波从 `WAIT_SYNC` 变为 `READY`。 | 同理，等待期间计为 stall。 |
| 指令冲突 | 在 `checkResources` 中比较所需资源与当前周期可用资源（例如：每个 CU 有 4 个 VALU，若已有 4 条 VALU 指令在执行管道中，则新的 VALU 指令不可发射）。 | 若不可发射，则当前周期记为 **bubble**（对应资源空闲时间）。 |

- **空泡统计**    - 每周期统计 `idleVALU`, `idleSALU`, `idleLDS`, `idleTEX`。  
  - `totalIdleCycles = Σ_cycle (idleVALU + idleSALU + …)`。  
  - 利用率：  
    $$
    \text{VALU Utilization} = 1 - \frac{\text{totalIdleVALU}}{\text{totalCycles} \times \text{NUM_VALU_PER_CU} \times \text{NUM_CU}}
    $$  

#### 3.3.5 Perfetto Trace 生成  

- 使用官方 **Perfetto C++ SDK**（需链接 `libperfetto.a`）。  - 定义两种 `TraceEvent` 类型：  

  | 事件名 | 字段 | 含义 |
|--------|------|------|
| `WaveExec` | `name: "WaveExec"`<br>`waveID: uint32`<br>`instrIdx: uint32`<br>`latency: uint32`<br>`resource: string` | 某波在某周期发射了一条指令。 |
| `Bubble`   | `name: "Bubble"`<br>`cycle: uint32`<br>`idleVALU: uint32`<br>`idleSALU: uint32`<br>`idleLDS: uint32`<br>`idleTEX: uint32` | 某周期出现的资源空闲。 |

- 伪码（每 tick 结束后调用）：  

  ```cpp
  void PerfettoCollector::flushTick(uint32_t cycle) {
      perfetto::TraceWriter trace = tracer_.NewTrace();
      auto* packet = trace.NewPacket();
      packet->set_timestamp(cycle * kCyclePeriodNs); // 假设 1 轨道 = 1 ns，可调

      // WaveExec events
      for (auto& ev : waveEventsThisTick) {
          auto* evPacket = packet->add_event();
          evPacket->set_name("WaveExec");
          evPacket->add_args()->set_name("waveID");
          evPacket->mutable_args(0)->set_uint_value(ev.waveID);
          evPacket->add_args()->set_name("instrIdx");
          evPacket->mutable_args(1)->set_uint_value(ev.instrIdx);
          evPacket->add_args()->set_name("latency");
          evPacket->mutable_args(2)->set_uint_value(ev.latency);
          evPacket->add_args()->set_name("resource");
          evPacket->mutable_args(3)->set_string_value(ev.resource);
      }

      // Bubble event      auto* bub = packet->add_event();
      bub->set_name("Bubble");
      bub->add_args()->set_name("cycle");
      bub->mutable_args(0)->set_uint_value(cycle);
      bub->add_args()->set_name("idleVALU");
      bub->mutable_args(1)->set_uint_value(idleVALU_);
      // …其它资源
  }
  ```

- 仿真结束后调用 `tracer_.Flush()` 并写入文件 `trace.pb`。  

---  

## 4. 输出与统计  | 输出项 | 说明 | 产生位置 |
|--------|------|----------|
| **总周期数** (`totalCycles`) | 从第一次发射到所有波 `FINISHED` 的墙钟时间（以模拟周期为单位）。 | `Scheduler::totalCycles_` |
| **波平均周期** (`avgWaveCycles`) | `totalCycles / totalWaves`（粗略指标）。 | `Statistics` |
| **波 stall 次数** (`waveStalls[wid]`) | 波因同步/资源而未被调度的周期数。 | `Statistics::recordStall(wid, cnt)` |
| **资源利用率** (`VALUUtil`, `SALUUtil`, …) | 如上公式所示。 | `Statistics::calcUtilization()` |
| **Perfetto trace** (`trace.pb`) | 可在 Perfetto UI 中打开，查看每条指令的发射时间、资源占用、空泡段落。 | `PerfettoCollector::writeTrace(outFile)` |
| **功能校验**（可选） | 比较 Host‑side CPU 执行的金标准结果与模拟内存的值，返回 `true/false`。 | `Validator::verify()` |
| **日志** (`sim.log`) | 每 tick 的简要信息（可调节日志级别），便于离线分析。 | `Logger` |  

> **注意**：模拟器不产生确切的硬件时钟（如实际 GPU 频率），只给出 **相对周期数**。若需要换算成实际时间，可乘以假设的时钟周期（例如 1 ns 对应 1 GHz），但这仅用于 **趋势对比**，不用于绝对性能预测。  

---  

## 5. 使用流程与示例  

### 5.1 编译  

```bash
# 假设源码位于 src/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# 生成 libcycle.so（动态库）以及测试程序 demo
```  

### 5.2 运行原始程序（带 hook）  ```bash
LD_PRELOAD=$(pwd)/libcycle.so ./my_original_app.out  # 原始程序不需修改
```

- 程序启动时会自动拦截 `hipLaunchKernelGGL`，调用仿真器。  - 仿真结束后，仍会调用原始 HIP 实现（若要**纯仿真**可在 hook 中直接 `return`，不调用原始函数）。  

### 5.3 查看结果  

- 终端会打印类似：  

  ```
  [CycleSim] Kernel 'my_kernel' launched.
  [CycleSim] Total waves: 1536
  [CycleSim] Simulated cycles: 8423100
  [CycleSim] VALU utilization: 68.4%
  [CycleSim] LDS utilization: 42.1%
  [CycleSim] Total bubble cycles: 1.23M
  [CycleSim] Perfetto trace written to trace.pb
  ```  

- 使用 Perfetto 查看：  

  ```bash
  # 安装 Perfetto UI（官方提供的网页版也可直接打开）
  perfetto --txt --out=trace.json trace.pb   # 转为 json  # 或者直接在浏览器打开 https://ui.perfetto.dev/ 并载入 trace.pb
  ```  

- 在 UI 中可以看到：  

  - 每条 `WaveExec` 事件的时间戳（波 ID、指令索引、延迟）。  
  - `Bubble` 段落显示哪些周期哪些资源空闲。  
  - 通过过滤 `waveID == 42` 可观察单个波的执行时间线。  

### 5.4 功能校验（可选）  

若希望在仿真后立即验证正确性：  ```cpp
// 在 hook 中
bool ok = Validator::verify(hostPtr, expectedSize);
if (!ok) {
    std::cerr << "[CycleSim] ERROR: result mismatch!\n";
    std::abort();
}
```  

---  

## 6. 性能与精度权衡  

| 方面 | 设计选择 | 影响 | 说明 |
|------|----------|------|------|
| **指令模型** | 固定延迟 + 资源集合（无流水线细节） | **低开销**，但可能错过某些指令交叉冲突（如指令合并） | 足以捕获主要的瓶颈（算术 vs 存储 vs 同步）。 |
| **波调度** | 轮询 + 简单资源检测（无动态优先级、无 warp‑level 调度器） | **实现简单**，开销 O(numWaves) 每 tick（可接受，因为波数通常 ≤ 十万） | 若需要更精细的调度可插入优先级队列。 |
| **内存模型** | 延迟固定（全局内存 400 徐秒，LDS 1 徐秒） | **可调参数**，不模拟缓存层次、银行冲突 | 对于以算术为主的 kernel 已足够；若要研究存储瓶颈可细化延迟表。 |
| **同步** | 计数器模型（`s_waitcnt`、`s_barrier`） | **精确** 适用于大多数 barrier/waitcnt 使用模式 | 不处理更复杂的 `__syncthreads()` 变种（如带条件的 barrier），但这些在 HIP 中极少见。 |
| **Perfetto 开销** | 每 tick 写入少量事件（约数十字节） | **可忽略**；若波数极大（>1M）可采用 **采样**（每 N tick 写一次） | 保证交互式分析流畅。 |
| **总体速度** | 在现代 X86‑64 上，约 10⁴ – 10⁵ wave/s（取决于指令密度） | **远快于** 全功能 RTL 模拟（后者往往每秒只有几百 wave） | 适合用于编译器迭代验证、算子库参数 sweep。 |  

> **结论**：该设计在 **功能正确**、**轻量** 和 **可观测** 之间达到了良好的平衡，适合作为编译器/算子库的 **快速反馈工具**。  

---  

*以上即为完整的设计方案，可直接复制保存为 `AMDGPU_Cycle_Simulator_Design.md` 使用。*
