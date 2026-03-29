
# AMDGPU轻量级Cycle模拟器设计方案

## 文档信息
- 版本: v1.0
- 日期: 2026-03-29
- 架构目标: AMD GCN/CDNA/RDNA系列
- 设计原则: 功能正确性优先，轻量级Cycle近似，编译器/算子优化导向

---

## 目录
1. [High Level Design](#high-level-design)
   - 1.1 设计目标与约束
   - 1.2 整体架构
   - 1.3 核心抽象层
   - 1.4 数据流设计
   - 1.5 Cycle统计与输出定义
2. [Low Level Design](#low-level-design)
   - 2.1 模块详细设计
   - 2.2 指令集解析
   - 2.3 Wave调度器
   - 2.4 内存模型
   - 2.5 Perfetto集成
   - 2.6 API设计
3. [实现路线图](#实现路线图)

---

## High Level Design

### 1.1 设计目标与约束

#### 核心目标
| 目标 | 优先级 | 说明 |
|------|--------|------|
| 功能正确性 | P0 | Host执行结果与真实GPU完全一致，通过校验 |
| Cycle近似精度 | P1 | 相对误差<15%，绝对误差可接受范围 |
| 轻量级 | P0 | 单核模拟速度>10K instr/s，内存<2GB |
| 优化分析能力 | P1 | 识别ALU/MEM/Latency/Dependency bound |
| 可视化 | P2 | Perfetto trace输出，直观时间线 |

#### 非目标（明确排除）
- 不模拟RTL级信号和时序
- 不模拟缓存一致性协议细节
- 不模拟PCIe传输延迟（仅模拟kernel执行）
- 不追求100% cycle精确（面向编译器优化而非硬件验证）

#### 关键设计约束
```
输入: 标准ELF .out文件（ROCm/HIP编译产出）
执行: Host x86_64 CPU执行，动态库hook拦截Runtime API
输出: 
  - 功能结果: 与GPU一致的内存状态
  - 性能数据: Total Cycles, IPC, Utilization, Bound Analysis
  - Trace: Perfetto格式可视化时间线
```

### 1.2 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Application                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   App Code  │  │   hipMalloc │  │   hipLaunchKernel(...)  │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Hook Layer (LD_PRELOAD)                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  libamdgpu_simulator_hook.so                                │  │
│  │  - intercept hipLaunchKernel                                │  │
│  │  - intercept memory APIs (hipMalloc, hipMemcpy)             │  │
│  │  - redirect to simulator                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Simulator Core                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ELF Loader   │  │ Kernel Config│  │   Execution Engine   │   │
│  │ - Parse .out │  │ - Grid/Block │  │   - Wave Pool        │   │
│  │ - Extract    │  │   Dimension  │  │   - Scheduler        │   │
│  │   .text/.rodata│ - Metadata   │  │   - Functional Model │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Instruction  │  │ Memory Model │  │   Cycle Counter      │   │
│  │ Decoder      │  │ - Global Mem │  │   - Per Wave         │   │
│  │ (SOP/VOP/DS) │  │ - LDS/Scratch│  │   - Per CU           │   │
│  │              │  │ - Cache Model│  │   - Global Timer     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output & Analysis                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Result       │  │ Statistics   │  │   Perfetto Trace     │   │
│  │ Validation   │  │ - IPC        │  │   - Wave Timeline    │   │
│  │ (memcmp)     │  │ - Occupancy  │  │   - Stall Events     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 核心抽象层

#### 1.3.1 执行模型抽象

```cpp
// 核心抽象：Wavefront（64线程SIMD执行单元）
struct Wavefront {
    uint32_t wave_id;           // 全局Wave ID
    uint32_t simd_id;           // 所属SIMD单元 (0-3 per CU)
    uint32_t cu_id;             // 所属CU ID
    uint32_t wg_id;             // Workgroup ID

    // 执行状态
    enum State { RUNNING, STALLED, BARRIER_WAIT, MEM_WAIT, EXIT } state;
    uint64_t pc;                // 当前程序计数器
    uint32_t exec_mask;         // 64位执行掩码（活跃线程）

    // 寄存器文件（功能模拟）
    VectorRegs vregs;           // VGPRs (每个线程256 x 32-bit)
    ScalarRegs sregs;           // SGPRs (每个Wave 128 x 32-bit)

    // Cycle统计
    uint64_t cycles_executed;   // 实际执行cycle
    uint64_t cycles_stalled;    // 空泡cycle
    StallReason last_stall;     // 上次stall原因
};

// Workgroup（Block）抽象
struct Workgroup {
    uint32_t wg_id;
    dim3 block_idx;
    uint32_t num_waves;
    std::vector<Wavefront*> waves;

    // 同步状态
    uint32_t waves_at_barrier;    // 到达barrier的wave计数
    bool barrier_reached;           // barrier是否触发

    // 共享内存
    LDSMemory lds;                  // 64KB Local Data Share
};

// Compute Unit（CU）抽象
struct ComputeUnit {
    uint32_t cu_id;
    static constexpr uint32_t NUM_SIMDS = 4;
    static constexpr uint32_t WAVES_PER_SIMD = 10; // 波前缓存深度

    // 资源状态
    uint32_t active_waves;
    std::array<SimdUnit, NUM_SIMDS> simds;

    // 执行单元
    ALUUnit alu_engine;
    MemUnit mem_engine;
    BranchUnit branch_engine;

    // Cycle计数器
    uint64_t current_cycle;
};
```

#### 1.3.2 指令抽象

```cpp
// 指令基类（ISA无关层）
struct Instruction {
    uint64_t addr;
    uint32_t raw_encoding;
    InstCategory category;

    // 延迟模型（Cycle近似关键）
    uint32_t latency;           // 结果产生延迟
    uint32_t throughput;        // 发射间隔（cycle）
    uint32_t issue_cycles;      // 执行所需cycle

    // 资源占用
    uint32_t alu_units;         // 占用的ALU单元数
    bool uses_mem;              // 是否访存
    bool is_branch;             // 是否分支
    bool is_barrier;            // 是否同步指令

    virtual void execute(Wavefront& wave, ExecContext& ctx) = 0;
};

// AMDGPU特定指令类型
enum class InstCategory {
    SOP1, SOP2, SOPK, SOPC, SOPP,    // Scalar ALU/Control
    SMEM,                             // Scalar Memory
    VOP1, VOP2, VOP3, VOPC, VOP3P,   // Vector ALU
    VINTRP,                           // Interpolation
    DS,                               // Data Share (LDS)
    MUBUF, MTBUF, MIMG, FLAT,         // Memory
    EXP,                              // Export (Pixel)
    LDS_DIRECT                        // LDS direct access
};
```

### 1.4 数据流设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     模拟执行数据流                                │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Kernel Launch
┌─────────┐    ┌─────────────┐    ┌─────────────────┐    ┌────────┐
│ hipLaunch│ -> │ ELF Loader │ -> │ Kernel Metadata │ -> │ Grid  │
│ Kernel() │    │ 解析.out   │    │ - arch info     │    │ Config│
│          │    │ - code/data│    │ - kernel desc   │    │       │
└─────────┘    └─────────────┘    └─────────────────┘    └───┬────┘
                                                              │
Phase 2: Wavefront Generation                                   ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Grid/Block Dim  │ -> │ Wavefront       │ -> │ Wave Pool       │
│ - blockIdx.xyz  │    │ Generator       │    │ (Ready Queue)   │
│ - threadIdx.xyz │    │ - 64 threads/wave│    │                 │
│                 │    │ - WG grouping   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Phase 3: Cycle-by-Cycle Execution
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Wave Scheduler  │ -> │ Instruction     │ -> │ Functional      │
│ (Round-Robin    │    │ Execution       │    │ Update          │
│  + Scoreboard)  │    │ - ALU/MEM/BRANCH│    │ - Regs/Mem      │
│                 │    │ - Latency tracking│   │ - Barrier sync  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│ Stall Detection │                            │ Cycle Counter   │
│ - RAW/WAW/WAR   │                            │ - Per Wave      │
│ - Mem latency   │                            │ - Per CU        │
│ - Barrier wait  │                            │ - Global        │
└─────────────────┘                            └─────────────────┘

Phase 4: Output & Trace
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Result Compare  │    │ Statistics      │    │ Perfetto Trace  │
│ (with GPU ref)  │    │ Aggregation     │    │ Generation      │
│                 │    │ - Total Cycles  │    │ - Wave timeline │
│                 │    │ - Bound analysis│    │ - Stall events  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1.5 Cycle统计与输出定义

#### 1.5.1 输出指标定义

```cpp
struct SimulationOutput {
    // === 核心性能指标 ===
    uint64_t total_cycles;              // 程序完成所需总cycle
    uint64_t total_instructions;        // 退休指令总数
    double ipc;                          // 每cycle指令数

    // === Wave级别统计 ===
    struct WaveStats {
        uint64_t cycles_active;          // 实际执行cycle
        uint64_t cycles_stalled;         // 空泡cycle
        uint64_t instructions_retired;

        // Stall breakdown
        uint64_t stall_raw_latency;      // 数据依赖等待
        uint64_t stall_mem_latency;      // 内存访问等待
        uint64_t stall_barrier;          // barrier同步等待
        uint64_t stall_resource;         // 资源冲突（如ALU busy）
        uint64_t stall_branch;           // 分支预测/执行等待
    };
    std::vector<WaveStats> per_wave_stats;

    // === CU级别统计 ===
    struct CUStats {
        uint64_t cycles_busy;            // 有wave在执行
        uint64_t cycles_idle;            // 完全空闲
        uint32_t max_active_waves;       // 最大同时活跃wave数
        float avg_occupancy;             // 平均occupancy (0-10 per SIMD)

        // 功能单元利用率
        float alu_utilization;           // ALU使用周期占比
        float mem_unit_utilization;     // 访存单元使用占比
        float issue_stall_rate;          // 发射槽空闲率
    };
    std::vector<CUStats> per_cu_stats;

    // === Bound分析（关键！）===
    struct BoundAnalysis {
        float alu_bound_pct;             // ALU计算受限时间占比
        float mem_bound_pct;             // 内存访问受限时间占比
        float latency_bound_pct;         // 指令延迟受限时间占比
        float barrier_bound_pct;         // 同步受限时间占比
        float occupancy_bound_pct;       // 并行度不足占比

        // 主导bound类型
        BoundType primary_bound;
        BoundType secondary_bound;
    } bound_analysis;

    // === 内存统计 ===
    struct MemStats {
        uint64_t global_loads;
        uint64_t global_stores;
        uint64_t lds_accesses;
        uint64_t cache_hits;
        uint64_t cache_misses;
        float cache_hit_rate;
        uint64_t mem_bytes_transferred;
    } mem_stats;
};
```

#### 1.5.2 关键问题回答

**Q: 输出是单个Wave的Cycle还是整体程序Cycle？**

A: **两者都需要，但核心指标是整体程序Cycle**。

```cpp
// 整体程序完成时间（关键指标）
uint64_t program_completion_cycle = max(wave.completion_cycle for all waves);

// 但Wave级别cycle统计用于分析
for each wave:
    wave.total_cycles = wave.completion_cycle - wave.launch_cycle
    wave.stall_cycles = sum(all stall reasons)
    wave.active_cycles = wave.total_cycles - wave.stall_cycles
```

**Q: 能否观察指令队列空泡或部件空泡？**

A: **是的，设计包含多级空泡检测**：

```cpp
enum class BubbleType {
    // 前端空泡
    FETCH_BUBBLE,           // 指令获取延迟
    DECODE_BUBBLE,          // 解码延迟

    // 发射空泡
    ISSUE_BUBBLE_NO_WAVE,   // 无就绪wave
    ISSUE_BUBBLE_RESOURCE,  // 功能单元忙
    ISSUE_BUBBLE_SCOREBOARD,// 数据依赖

    // 执行空泡
    EXEC_BUBBLE_MEM_WAIT,   // 等待内存响应
    EXEC_BUBBLE_BARRIER,    // barrier等待
    EXEC_BUBBLE_SLEEP,      // sleep指令

    // 后端空泡
    RETIRE_BUBBLE_ORDER     // 顺序退休限制
};

// 在每个cycle记录
struct CycleSnapshot {
    uint64_t cycle;
    uint32_t cu_id;
    uint32_t simd_id;
    BubbleType bubble_type;
    uint32_t stalled_wave_id;
    InstType waiting_for_inst;  // 等待的指令类型
};
```

**Q: 是否可以利用Perfetto观察Cycle时间线？**

A: **是的，这是核心功能**。设计包含完整的Perfetto trace生成：

```cpp
// Perfetto trace事件类型
enum TraceEvent {
    WAVE_SCHEDULE,      // Wave开始执行
    WAVE_STALL,         // Wave进入stall状态
    WAVE_RESUME,        // Wave恢复执行
    WAVE_COMPLETE,      // Wave完成

    INST_ISSUE,         // 指令发射
    INST_RETIRE,        // 指令退休

    MEM_REQUEST,        // 内存请求发出
    MEM_RESPONSE,       // 内存响应返回

    BARRIER_ENTER,      // 进入barrier
    BARRIER_EXIT,       // 退出barrier

    BUBBLE_DETECTED     // 检测到空泡
};

// 生成Perfetto JSON或protobuf格式
void export_perfetto_trace(const std::string& filename);
```

---

## Low Level Design

### 2.1 模块详细设计

#### 2.1.1 ELF Loader模块

```cpp
class ElfLoader {
public:
    struct LoadedKernel {
        std::string kernel_name;

        // 代码段
        std::vector<uint8_t> code_section;      // .text
        uint64_t code_entry_point;

        // 数据段
        std::vector<uint8_t> rodata;            // 常量数据
        std::vector<uint8_t> data;              // 可写数据

        // AMDGPU特定元数据
        amd_kernel_code_t kernel_desc;          // AMD GPU内核描述符
        uint32_t arch_version;                  // gfx908, gfx90a, etc.

        // 资源需求
        uint32_t vgpr_count;
        uint32_t sgpr_count;
        uint32_t lds_size;
        uint32_t stack_size;

        // 配置
        dim3 grid_dim;
        dim3 block_dim;
        size_t shared_mem_bytes;
    };

    LoadedKernel load(const std::string& elf_path, const std::string& kernel_name);

private:
    void parse_amd_metadata(const ELFIO::elfio& elf, LoadedKernel& kernel);
    void extract_code_object(const ELFIO::elfio& elf, LoadedKernel& kernel);
};
```

#### 2.1.2 指令解码器

```cpp
class InstructionDecoder {
public:
    // 解码单条指令
    std::unique_ptr<Instruction> decode(uint64_t pc, uint32_t encoding);

    // 批量解码代码段
    std::vector<std::unique_ptr<Instruction>> decode_section(
        const uint8_t* code, size_t size, uint64_t base_addr);

private:
    // SOP2格式解码（示例）
    std::unique_ptr<Instruction> decode_sop2(uint32_t encoding);
    std::unique_ptr<Instruction> decode_vop2(uint32_t encoding);
    std::unique_ptr<Instruction> decode_ds(uint32_t encoding);
    std::unique_ptr<Instruction> decode_mubuf(uint32 encoding);

    // 延迟模型数据库（架构相关）
    LatencyModel latency_db_;
};

// 具体指令实现示例
class VOP2Instruction : public Instruction {
public:
    VOP2Opcode opcode;
    uint8_t vdst;       // 目的VGPR
    uint8_t vsrc0, vsrc1; // 源VGPR

    void execute(Wavefront& wave, ExecContext& ctx) override {
        // 功能模拟：执行实际计算
        for (int lane = 0; lane < 64; ++lane) {
            if (wave.exec_mask & (1ULL << lane)) {
                float src0 = wave.vregs.read(lane, vsrc0);
                float src1 = wave.vregs.read(lane, vsrc1);
                float result = alu_op(src0, src1, opcode);
                wave.vregs.write(lane, vdst, result);
            }
        }

        // 更新PC
        wave.pc += 4;
    }

private:
    float alu_op(float a, float b, VOP2Opcode op);
};
```

#### 2.1.3 Wave调度器（核心模块）

```cpp
class WaveScheduler {
public:
    WaveScheduler(uint32_t num_cus);

    // 主调度循环
    void run_to_completion();

    // 添加待执行wave
    void submit_wavefronts(const std::vector<Wavefront*>& waves);

    // 获取当前cycle
    uint64_t current_cycle() const { return global_cycle_; }

private:
    // 每cycle执行
    void cycle();

    // 调度策略
    Wavefront* select_wave_for_dispatch(CU& cu);
    Wavefront* select_wave_for_issue(SimdUnit& simd);

    // 状态更新
    void update_wave_state(Wavefront& wave);
    void handle_memory_response();
    void handle_barrier_sync();

    // 数据结构
    std::vector<ComputeUnit> cus_;
    std::queue<Wavefront*> pending_waves_;      // 等待分配的wave
    uint64_t global_cycle_;

    // 记分板（数据依赖跟踪）
    Scoreboard scoreboard_;

    // 事件队列（用于异步事件如内存响应）
    std::priority_queue<ScheduledEvent> event_queue_;
};

// 记分板实现
class Scoreboard {
public:
    // 检查指令是否可以发射（无RAW/WAW冲突）
    bool can_issue(const Wavefront& wave, const Instruction& inst);

    // 标记指令发射，记录结果产生时间
    void mark_issued(Wavefront& wave, const Instruction& inst, uint64_t cycle);

    // 更新cycle，清除已完成的依赖
    void advance_cycle(uint64_t cycle);

private:
    struct PendingResult {
        uint8_t reg_type;   // 0=SGPR, 1=VGPR
        uint8_t reg_num;
        uint64_t ready_cycle;
    };

    std::unordered_map<uint32_t, std::vector<PendingResult>> wave_pending_;
};
```

#### 2.1.4 内存模型

```cpp
class MemoryModel {
public:
    // 初始化内存空间
    void allocate_global(size_t size, uint64_t gpu_va);
    void allocate_lds(uint32_t wg_id, size_t size);

    // 内存访问接口
    MemResponse load(uint64_t addr, size_t size, MemSpace space);
    void store(uint64_t addr, const void* data, size_t size, MemSpace space);

    // 获取访问延迟（cycle数）
    uint32_t get_latency(uint64_t addr, MemSpace space, bool is_load);

private:
    // 内存空间
    std::unordered_map<uint64_t, std::vector<uint8_t>> global_mem_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> lds_mem_;  // per WG

    // 缓存模型（简化）
    SimpleCache l1_cache_;

    // 延迟配置
    struct LatencyConfig {
        uint32_t lds_latency;
        uint32_t l1_hit_latency;
        uint32_t l1_miss_latency;
        uint32_t global_latency_base;
    } config_;
};

// 异步内存请求（用于模拟延迟）
struct MemRequest {
    uint64_t issue_cycle;
    uint64_t ready_cycle;
    uint32_t wave_id;
    uint64_t addr;
    bool is_load;
    std::vector<uint8_t> data;  // for store
};
```

### 2.2 指令集解析

#### 2.2.1 AMDGPU指令格式支持

```cpp
// 支持的指令集架构版本
enum class ArchVersion {
    GFX803,     // Polaris
    GFX900,     // Vega
    GFX906,     // Vega 20
    GFX908,     // MI100 (CDNA)
    GFX90A,     // MI200 (CDNA2)
    GFX940,     // MI300 (CDNA3)
    GFX1030,    // RDNA2
    GFX1100     // RDNA3
};

// 指令解码表（部分示例）
const std::unordered_map<uint8_t, DecodeFunc> decode_table = {
    // SOP2: Scalar ALU 2输入
    {0x00, [](uint32_t e) { return decode_sop2<SOP2Opcode::S_ADD_U32>(e); }},
    {0x01, [](uint32_t e) { return decode_sop2<SOP2Opcode::S_SUB_U32>(e); }},
    {0x02, [](uint32_t e) { return decode_sop2<SOP2Opcode::S_ADD_I32>(e); }},

    // SOPP: Scalar Control
    {0xBF, decode_sopp},

    // SMEM: Scalar Memory
    {0xC0, decode_smem},

    // VOP2: Vector ALU 2输入
    {0x00, [](uint32_t e) { return decode_vop2<VOP2Opcode::V_CNDMASK_B32>(e); }},
    {0x01, [](uint32_t e) { return decode_vop2<VOP2Opcode::V_ADD_F32>(e); }},
    {0x02, [](uint32_t e) { return decode_vop2<VOP2Opcode::V_SUB_F32>(e); }},
    {0x03, [](uint32_t e) { return decode_vop2<VOP2Opcode::V_SUBREV_F32>(e); }},
    {0x04, [](uint32_t e) { return decode_vop2<VOP2Opcode::V_MUL_LEGACY_F32>(e); }},

    // VOP3: Vector ALU 3输入（高延迟复杂操作）
    {0xD1, decode_vop3},

    // DS: Data Share (LDS)
    {0xD8, decode_ds},

    // MUBUF: Memory Buffer
    {0xDC, decode_mubuf},

    // MTBUF: Typed Memory Buffer
    {0xD9, decode_mtbuf},

    // FLAT: Flat Memory（全局内存）
    {0xDD, decode_flat},
};
```

#### 2.2.2 延迟模型配置

```cpp
// 架构特定的延迟参数（以CDNA2/GFX90A为例）
struct ArchLatencyModel {
    // ALU延迟
    uint32_t alu_fma_latency = 4;           // FMA操作
    uint32_t alu_add_latency = 2;           // 简单整数/浮点加
    uint32_t alu_mul_latency = 4;           // 乘法
    uint32_t alu_transcendental_latency = 16; // sin/cos/log等

    // 内存延迟（cycle）
    uint32_t lds_latency = 24;              // LDS访问
    uint32_t l1_hit_latency = 64;           // L1命中
    uint32_t l1_miss_latency = 400;         // L1未命中（到L2）
    uint32_t l2_miss_latency = 800;         // L2未命中（到HBM）

    // 发射吞吐（每SIMD每cycle）
    uint32_t alu_issues_per_cycle = 1;      // 大多数ALU
    uint32_t mem_issues_per_cycle = 1;      // 内存指令
    uint32_t branch_issues_per_cycle = 1;   // 分支指令

    // 特殊指令
    uint32_t barrier_latency = 10;          // barrier同步开销
    uint32_t sleep_latency_base = 64;       // s_sleep基础延迟
};

// 根据指令类型获取延迟
uint32_t get_instruction_latency(const Instruction& inst, ArchVersion arch) {
    static std::unordered_map<ArchVersion, ArchLatencyModel> models;

    const auto& model = models[arch];

    switch (inst.category) {
        case InstCategory::VOP2:
            if (is_transcendental(inst.opcode)) 
                return model.alu_transcendental_latency;
            if (is_multiplication(inst.opcode)) 
                return model.alu_mul_latency;
            return model.alu_add_latency;

        case InstCategory::VOP3:
            return model.alu_fma_latency;

        case InstCategory::DS:
            return model.lds_latency;

        case InstCategory::FLAT:
        case InstCategory::MUBUF:
            // 实际延迟由内存模型动态计算
            return 0; // 由内存系统返回实际值

        case InstCategory::SOPP:
            if (inst.opcode == SOPPOpcode::S_BARRIER)
                return model.barrier_latency;
            return 1;

        default:
            return 4; // 默认保守估计
    }
}
```

### 2.3 Wave调度器详细设计

#### 2.3.1 调度算法

```cpp
void WaveScheduler::cycle() {
    global_cycle_++;

    // 1. 处理异步事件（内存响应等）
    process_async_events();

    // 2. 每个CU独立调度
    for (auto& cu : cus_) {
        // 2.1 尝试分配新wave到空闲SIMD
        dispatch_waves_to_cu(cu);

        // 2.2 每个SIMD选择并发射指令
        for (auto& simd : cu.simds) {
            if (auto* wave = select_wave_for_issue(simd)) {
                if (auto* inst = fetch_instruction(*wave)) {
                    if (scoreboard_.can_issue(*wave, *inst)) {
                        if (try_execute_instruction(cu, simd, *wave, *inst)) {
                            scoreboard_.mark_issued(*wave, *inst, global_cycle_);
                            record_issue_event(*wave, *inst);
                        } else {
                            // 资源冲突，标记stall
                            record_resource_stall(*wave, *inst);
                            wave->state = Wavefront::STALLED;
                            wave->last_stall = StallReason::RESOURCE_CONFLICT;
                        }
                    } else {
                        // 数据依赖，标记stall
                        record_dependency_stall(*wave, *inst);
                        wave->state = Wavefront::STALLED;
                        wave->last_stall = StallReason::DATA_DEPENDENCY;
                    }
                }
            } else {
                // 无就绪wave，SIMD空泡
                record_simd_bubble(cu.cu_id, simd.id, BubbleType::ISSUE_BUBBLE_NO_WAVE);
            }
        }

        // 2.3 更新stall计数
        update_stall_stats(cu);
    }

    // 3. 检查完成状态
    check_completion();
}

Wavefront* WaveScheduler::select_wave_for_issue(SimdUnit& simd) {
    // 轮询策略，跳过stall的wave
    for (int i = 0; i < simd.waves.size(); ++i) {
        simd.round_robin_idx = (simd.round_robin_idx + 1) % simd.waves.size();
        auto* wave = simd.waves[simd.round_robin_idx];

        if (wave->state == Wavefront::RUNNING) {
            return wave;
        }
    }
    return nullptr;
}

bool WaveScheduler::try_execute_instruction(CU& cu, SimdUnit& simd, 
                                          Wavefront& wave, Instruction& inst) {
    switch (inst.category) {
        case InstCategory::VOP1:
        case InstCategory::VOP2:
        case InstCategory::VOP3:
            if (cu.alu_engine.is_available()) {
                cu.alu_engine.schedule(wave, inst, global_cycle_);
                return true;
            }
            return false;

        case InstCategory::DS:
        case InstCategory::FLAT:
        case InstCategory::MUBUF:
            if (cu.mem_engine.is_available()) {
                // 内存指令异步执行
                auto latency = memory_model_.get_latency(...);
                cu.mem_engine.schedule_async(wave, inst, global_cycle_, latency);
                return true;
            }
            return false;

        case InstCategory::SOPP:
            if (inst.opcode == SOPPOpcode::S_BARRIER) {
                handle_barrier(wave);
                return true;
            }
            // 其他scalar指令直接执行
            inst.execute(wave, ctx_);
            return true;

        default:
            inst.execute(wave, ctx_);
            return true;
    }
}
```

#### 2.3.2 Barrier同步实现

```cpp
void WaveScheduler::handle_barrier(Wavefront& wave) {
    auto& wg = workgroups_[wave.wg_id];

    // 标记当前wave到达barrier
    wave.state = Wavefront::BARRIER_WAIT;
    wg.waves_at_barrier++;

    record_barrier_enter(wave);

    // 检查是否所有wave都到达
    if (wg.waves_at_barrier == wg.num_waves) {
        // 释放所有wave
        for (auto* w : wg.waves) {
            w->state = Wavefront::RUNNING;
            w->pc += 4; // 跳过barrier指令
        }
        wg.waves_at_barrier = 0;
        record_barrier_exit(wg.wg_id, global_cycle_);
    }
}
```

### 2.4 Perfetto集成

#### 2.4.1 Trace生成器

```cpp
class PerfettoTraceGenerator {
public:
    void init(const std::string& session_name);
    void finalize(const std::string& output_path);

    // Wave生命周期事件
    void emit_wave_create(uint32_t wave_id, uint64_t ts, uint32_t wg_id);
    void emit_wave_start(uint32_t wave_id, uint64_t ts);
    void emit_wave_stall(uint32_t wave_id, uint64_t ts, StallReason reason);
    void emit_wave_resume(uint32_t wave_id, uint64_t ts);
    void emit_wave_complete(uint32_t wave_id, uint64_t ts);

    // 指令事件
    void emit_inst_issue(uint32_t wave_id, uint64_t ts, uint64_t pc, 
                        InstCategory cat);
    void emit_inst_retire(uint32_t wave_id, uint64_t ts, uint64_t pc);

    // CU/SIMD利用率
    void emit_cu_active(uint32_t cu_id, uint64_t ts, bool active);
    void emit_simd_occupancy(uint32_t cu_id, uint32_t simd_id, 
                            uint64_t ts, uint32_t num_waves);

    // 内存事件
    void emit_mem_request(uint32_t wave_id, uint64_t ts, uint64_t addr, 
                         MemSpace space);
    void emit_mem_complete(uint32_t wave_id, uint64_t ts, uint64_t latency);

    // 空泡事件（关键！）
    void emit_bubble(uint32_t cu_id, uint32_t simd_id, uint64_t ts, 
                    BubbleType type, const std::string& details);

private:
    // 使用Perfetto SDK或生成JSON格式
    void write_json_event(const std::string& category, 
                         const std::string& name,
                         const std::string& phase,  // B=begin, E=end, X=complete
                         uint64_t ts,
                         uint32_t tid,  // 映射到wave_id或cu_id
                         const json& args);

    std::ofstream trace_file_;
    uint64_t start_ts_;
};
```

#### 2.4.2 Trace可视化示例

生成的Perfetto trace将显示：

```
时间线视图（Chrome Trace Viewer）：
─────────────────────────────────────────────────────────────►

Wave 0:  [EXEC][EXEC][STALL:MEM][STALL:MEM][EXEC][EXEC][DONE]
Wave 1:  [EXEC][EXEC][EXEC][STALL:RAW][EXEC][EXEC][DONE]
Wave 2:  [EXEC][BARRIER_WAIT________________][EXEC][DONE]
Wave 3:  [EXEC][EXEC][BARRIER_WAIT________________][EXEC][DONE]

CU 0 SIMD0:  Occupancy: 2 | 2 | 2 | 1 | 1 | 2 | 2 | 0
CU 0 SIMD1:  Occupancy: 2 | 2 | 2 | 2 | 2 | 2 | 1 | 0

内存请求:  |----Req----|----------Latency----------|Resp|
Bubble:       [ISSUE_BUBBLE_RESOURCE]  
```

### 2.5 API设计

#### 2.5.1 用户接口

```cpp
// C++ API
namespace amdgpu_sim {

class Simulator {
public:
    struct Config {
        ArchVersion arch = ArchVersion::GFX90A;
        uint32_t num_cus = 80;           // MI200默认
        uint32_t waves_per_simd = 10;
        bool enable_perfetto = true;
        std::string perfetto_output = "trace.json";
        bool validate_results = true;
    };

    explicit Simulator(const Config& config);

    // 加载kernel
    void load_kernel(const std::string& elf_path, 
                    const std::string& kernel_name);

    // 设置kernel参数（与HIP对应）
    void set_grid_dim(dim3 grid);
    void set_block_dim(dim3 block);
    void set_kernel_arg(size_t idx, size_t size, const void* data);

    // 分配设备内存（模拟）
    DevicePtr malloc(size_t size);
    void memcpy_h2d(DevicePtr dst, const void* src, size_t size);
    void memcpy_d2h(void* dst, DevicePtr src, size_t size);

    // 执行模拟
    SimulationOutput run();

    // 获取结果
    const SimulationOutput& get_output() const;
    void export_perfetto(const std::string& path);
};

// Python绑定（pybind11）
// usage: import amdgpu_sim; sim = amdgpu_sim.Simulator(...)

} // namespace amdgpu_sim
```

#### 2.5.2 Hook层实现

```cpp
// libamdgpu_simulator_hook.so
// 通过LD_PRELOAD拦截HIP Runtime API

extern "C" {

// 拦截kernel启动
hipError_t hipLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                          void** args, size_t sharedMem, hipStream_t stream) {
    // 获取当前ELF和kernel名称
    auto [elf_path, kernel_name] = get_kernel_info(func);

    // 初始化模拟器（单例）
    static Simulator sim(get_config_from_env());

    // 加载并配置kernel
    sim.load_kernel(elf_path, kernel_name);
    sim.set_grid_dim(gridDim);
    sim.set_block_dim(blockDim);

    // 设置参数（从args解析）
    for (int i = 0; args[i] != nullptr; ++i) {
        sim.set_kernel_arg(i, arg_sizes[i], args[i]);
    }

    // 执行模拟
    auto output = sim.run();

    // 输出统计
    if (getenv("AMDGPU_SIM_VERBOSE")) {
        print_statistics(output);
    }

    // 结果验证（如果配置了参考数据）
    if (getenv("AMDGPU_SIM_VALIDATE")) {
        validate_with_reference(output);
    }

    return hipSuccess;
}

// 拦截内存分配
hipError_t hipMalloc(void** ptr, size_t size) {
    *ptr = get_simulator().malloc(size);
    return hipSuccess;
}

// 拦截内存拷贝
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    switch (kind) {
        case hipMemcpyHostToDevice:
            get_simulator().memcpy_h2d((DevicePtr)dst, src, size);
            break;
        case hipMemcpyDeviceToHost:
            get_simulator().memcpy_d2h(dst, (DevicePtr)src, size);
            break;
        // ...
    }
    return hipSuccess;
}

} // extern "C"
```

---

## 实现路线图

### Phase 1: 基础框架（4周）
- [ ] ELF解析器（支持基本代码段提取）
- [ ] 基础指令解码器（SOP/VOP2/VOP3）
- [ ] 简单Wave调度器（无依赖跟踪）
- [ ] 标量内存模型
- [ ] 基础Perfetto trace输出

### Phase 2: 功能完整（4周）
- [ ] 完整指令集支持（DS/MUBUF/FLAT/EXP）
- [ ] 记分板依赖跟踪
- [ ] LDS/共享内存模型
- [ ] Barrier同步实现
- [ ] 异步内存延迟模拟
- [ ] Hook层实现

### Phase 3: 精度优化（3周）
- [ ] 架构特定延迟模型（CDNA2/RDNA3）
- [ ] 缓存模型（L1/L2近似）
- [ ] Wave调度策略优化
- [ ] 分支预测模型
- [ ] 与真实GPU结果校准

### Phase 4: 分析工具（3周）
- [ ] Bound分析算法
- [ ] 热点kernel识别
- [ ] 指令级分析（瓶颈指令定位）
- [ ] Roofline模型集成
- [ ] 可视化Dashboard

---

## 附录

### A. 关键设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 执行模型 | 解释执行 vs 二进制翻译 | 解释执行 | 轻量级，跨平台，易调试 |
| 调度粒度 | Wave级 vs Warp级 | Wave级 | AMDGPU原生是64-thread Wave |
| 内存模型 | 功能级 vs 时序精确 | 功能级+延迟近似 | 平衡精度与速度 |
| 依赖跟踪 | 记分板 vs  Tomasulo | 记分板 | 简单，足够用于stall分析 |
| Trace格式 | Perfetto vs 自定义 | Perfetto | 成熟工具链，可视化强大 |

### B. 参考文档
- AMD GCN ISA Architecture Manual
- AMD CDNA2 Instruction Set Architecture
- ROCm Compiler Infrastructure (LLVM AMDGPU Backend)
- Perfetto Trace Format Specification

### C. 术语对照

| 术语 | 英文 | 说明 |
|------|------|------|
| 波前 | Wavefront | 64线程SIMD执行单元 |
| 计算单元 | CU (Compute Unit) | 包含4个SIMD的硬件单元 |
| 本地数据共享 | LDS | 类似CUDA Shared Memory |
| 标量通用寄存器 | SGPR | 每Wave的标量寄存器 |
| 向量通用寄存器 | VGPR | 每线程的向量寄存器 |
| 工作项 | Work-item | 等同于CUDA Thread |
| 工作组 | Workgroup | 等同于CUDA Block |

---

**文档结束**

*本设计文档由AI生成，基于AMDGPU架构公开文档和通用GPU模拟器设计原则。*
*实际实现需参考最新AMD官方ISA手册和ROCm文档。*

