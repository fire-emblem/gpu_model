# AMD GPU 轻量级 Cycle 模拟器设计方案

## 一、设计目标与概述

### 1.1 设计目标

本设计方案旨在构建一个**轻量级**的 AMD GPU cycle 模拟器，专注于编译器和算子库的指令与算法优化分析。与传统 RTL 仿真不同，本模拟器追求**可接受的精度**与**合理的性能**之间的平衡，主要用于：

- **编译器优化验证**：验证编译器生成的指令序列是否达到预期性能
- **算子库性能分析**：分析算子实现中的瓶颈和优化空间
- **架构特征研究**：理解 AMD CDNA/CDNA2 架构的 wave 调度和执行模型

### 1.2 核心设计原则

```
┌─────────────────────────────────────────────────────────────────┐
│                     设计原则金字塔                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌───────────┐                              │
│                      │ 轻量级    │                              │
│                      │ (核心)    │                              │
│                      └─────┬─────┘                              │
│                      ┌─────┴─────┐                              │
│                      │ 功能正确  │                              │
│                      │ + 性能    │                              │
│                      └─────┬─────┘                              │
│                      ┌─────┴─────┐                              │
│                      │ 可扩展   │                              │
│                      │ + 可视化  │                              │
│                      └───────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、系统架构概览

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           AMD GPU Cycle Simulator                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Loader     │───▶│  Decoder     │───▶│  Scheduler   │───▶│  Simulator   │  │
│  │   Module     │    │  Module      │    │  Module      │    │  Core        │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                                      │           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │           │
│  │   Runtime    │    │  Analysis    │    │  Perfetto    │◀──────────┘           │
│  │   Hook       │    │  Engine      │    │  Tracer      │                        │
│  └──────────────┘    └──────────────┘    └──────────────┘                        │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Output Artifacts                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ Cycle      │  │ Wave       │  │ Timeline   │  │ Bottleneck │                │
│  │ Summary    │  │ Schedule   │  │ Trace      │  │ Report     │                │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   .out File          Host Execution          IR Representation      Cycle Model│
│  ┌─────────┐        ┌─────────┐             ┌─────────┐           ┌─────────┐ │
│  │ ELF     │  ───▶  │ Runtime │  ───▶       │ Wave    │  ───▶    │ Cycle   │ │
│  │ Parser  │        │ Hook    │             │ IR      │          │ Compute │ │
│  └────┬────┘        └────┬────┘             └────┬────┘           └────┬────┘ │
│       │                  │                       │                    │       │
│       ▼                  ▼                       ▼                    ▼       │
│  ┌─────────┐        ┌─────────┐             ┌─────────┐           ┌─────────┐ │
│  │ Code    │        │ Kernel  │             │Instr    │           │ Wave   │ │
│  │ Segment │        │ Launch  │             │ Seq     │           │ Sched  │ │
│  └─────────┘        └─────────┘             └─────────┘           └─────────┘ │
│                                                                                 │
│  ┌─────────┐        ┌─────────┐             ┌─────────┐           ┌─────────┐ │
│  │ Data    │        │ Memory  │             │ SGPR/   │           │ Memory  │ │
│  │ Segment │        │ Access  │             │ VGPR    │           │ Latency │ │
│  └─────────┘        └─────────┘             └─────────┘           └─────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、High Level 设计

### 3.1 模块职责划分

| 模块 | 职责 | 关键接口 |
|------|------|----------|
| **Loader Module** | 解析 AMD GPU 二进制格式，提取代码段、数据段、metadata | `load_elf()`, `extract_metadata()` |
| **Runtime Hook** | 通过 LD_PRELOAD 拦截 ROCm runtime API | `hipModuleLaunchKernel()`, `hipMemcpy()` |
| **Decoder Module** | 将 GCN/AMDGCN 指令字节码解析为指令对象 | `decode_instruction()`, `build_instr_seq()` |
| **Scheduler Module** | 管理 wave 的创建、分发和调度 | `create_waves()`, `dispatch_wave()`, `wave_yield()` |
| **Simulator Core** | 执行指令的 cycle-accurate 模拟 | `execute_wave()`, `simulate_cycle()` |
| **Analysis Engine** | 性能瓶颈分析和统计 | `analyze_bottleneck()`, `compute_stats()` |
| **Perfetto Tracer** | 生成性能追踪数据 | `trace_event()`, `generate_trace()` |

### 3.2 执行模型

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AMD GPU Execution Model                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Host                                                                  GPU     │
│  ┌───────┐                                                              ┌─────┐  │
│  │ CPU   │ ────── hipModuleLaunchKernel() ─────────────────────────▶  │ CU  │  │
│  └───────┘          (Intercepted by Hook)                              └──┬──┘  │
│                                                                          │     │
│     │                                                                     │     │
│     │              Grid = {DimX, DimY, DimZ}                             │     │
│     │                                                                     ▼     │
│     │              Block = {DimX, DimY, DimZ}                     ┌─────────┐ │
│     │              Total Threads = Grid × Block                   │ Wave    │ │
│     │                                                       64     │ Front   │ │
│     │              Wave 0 ──▶  Thread 0-63                         │ End     │ │
│     │              Wave 1 ──▶  Thread 64-127          ──────────▶ └────┬────┘ │
│     │              Wave 2 ──▶  Thread 128-191                         │      │
│     │              ...                                                 │      │
│     │              Each wave executes SIMD-like,                      ▼      │
│     │              one instruction at a time                  ┌───────────┐  │
│     │                                                    ┌──▶  │ Vector    │  │
│     │              Synchronization:                   │    │ Unit      │  │
│     │              - __syncthreads()                   │    └─────┬─────┘  │
│     │              - __threadfence()                   │          │         │
│     │              - wave shift / rotate               │          ▼         │
│     │                                                    │    ┌───────────┐  │
│     │              Wave Yield (Async):                 └───▶│ Scalar    │  │
│     │              - s_sleep                              │    │ Unit      │  │
│     │              - s_waitcnt                           │    └───────────┘  │
│     │              - memory instruction                                               │
│     └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Cycle 模拟层次

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Cycle Simulation Layers                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Layer 4: System Level (最高层)                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Total Cycle = Σ(all waves on all CUs) + Memory System Latency         │   │
│  │  Output: End-to-end execution time, timeline visualization             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  Layer 3: Wave Level                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Wave Cycle = Σ(instr_cycles) + scheduling_overhead + contention       │   │
│  │  Output: Per-wave execution time, wave utilization                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  Layer 2: Instruction Level                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Instr Cycle = base_latency + operand_cycles + pipeline_cycles         │   │
│  │  Output: Per-instruction latency, throughput                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  Layer 1: Unit Level (最底层)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Vector Unit: 4-cycle base (VADD, VMUL), 16-cycle (VALU complex)       │   │
│  │  Scalar Unit: 1-cycle base, 4-cycle (SALU complex)                     │   │
│  │  Memory Unit: 4-cycle base + 400-cycle global memory                   │   │
│  │  LDS Unit: 4-cycle base + 32-cycle bank conflict                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、Low Level 设计

### 4.1 核心数据结构

#### 4.1.1 指令对象

```cpp
// 指令对象 - 模拟 AMD GPU 指令
struct Instruction {
    // 基础信息
    uint32_t        opcode;           // 操作码
    uint32_t        encoding;         // 原始编码
    InstructionType type;             // 指令类型
    
    // 操作数
    std::vector<Operand> dst;         // 目的操作数
    std::vector<Operand> src;         // 源操作数
    
    // 执行属性
    uint32_t        vgpr_reads;       // VGPR 读取数量
    uint32_t        vgpr_writes;      // VGPR 写入数量
    uint32_t        sgpr_reads;       // SGPR 读取数量
    uint32_t        sgpr_writes;      // SGPR 写入数量
    bool            is_memory;        // 是否内存指令
    bool            is_branch;        // 是否分支指令
    bool            is_vector;        // 是否向量指令
    
    // Cycle 信息
    uint32_t        base_latency;     // 基础延迟
    uint32_t        throughput;       // 吞吐量 (cycle)
    ExecUnit        exec_unit;        // 执行单元
};

// 执行单元枚举
enum class ExecUnit : uint8_t {
    VALU,  // Vector Arithmetic Logic Unit
    SALU,  // Scalar Arithmetic Logic Unit
    VMEM,  // Vector Memory
    SMEM,  // Scalar Memory
    LDS,   // Local Data Share
    GDS,   // Global Data Share
    BRANCH // Branch Unit
};
```

#### 4.1.2 Wave 对象

```cpp
// Wave 对象 - 代表一组 64 个线程的执行上下文
struct Wave {
    uint32_t            wave_id;          // Wave 全局 ID
    uint32_t            wave_in_block;    // Block 内的 wave 序号
    uint32_t            block_id;         // 所属 Block ID
    uint32_t            grid_id;          // 所属 Grid ID
    
    // 寄存器状态
    std::vector<uint32_t> vgpr_file;     // 向量通用寄存器文件
    std::vector<uint32_t> sgpr_file;     // 标量通用寄存器文件
    std::vector<uint32_t> pc_stack;      // PC 栈 (用于函数调用)
    
    // 执行状态
    uint32_t            pc;               // 程序计数器
    WaveState           state;            // 当前状态
    uint64_t            cycles;           // 已执行 cycle 数
    
    // 指令流
    std::vector<std::shared_ptr<Instruction>> instr_stream;
    uint32_t            current_instr_idx;
    
    // 资源追踪
    std::set<ExecUnit>  occupied_units;   // 当前占用的执行单元
    uint64_t            last_sched_cycle; // 上次调度时间
    
    // 同步原语状态
    uint32_t            barrier_count;    // 同步屏障计数
    bool                at_barrier;       // 是否在 barrier 处
};

// Wave 状态枚举
enum class WaveState : uint8_t {
    READY,      // 就绪，可被调度
    EXECUTING,  // 正在执行
    WAITING,    // 等待资源或同步
    YIELDED,    // 主动让出 (s_sleep)
    COMPLETED   // 执行完成
};
```

#### 4.1.3 CU (Compute Unit) 对象

```cpp
// Compute Unit - 模拟硬件 CU
struct ComputeUnit {
    uint32_t            cu_id;            // CU ID
    
    // 硬件配置
    uint32_t            num_waveslots;    // 最大并发 wave 数 (通常 8)
    uint32_t            num_valus;        // VALU 数量 (通常 4)
    uint32_t            num_salus;        // SALU 数量 (通常 1)
    uint32_t            num_vmem;         // Vector Memory 单元数
    uint32_t            num_lds;          // LDS 单元数
    
    // 资源状态
    std::vector<std::shared_ptr<Wave>> active_waves;  // 当前活跃的 waves
    std::queue<std::shared_ptr<Wave>> wave_queue;      // 等待队列
    
    // 执行单元状态
    std::vector<bool>   valu_available;
    bool                salu_available;
    std::vector<bool>   vmem_available;
    bool                lds_available;
    
    // 性能统计
    uint64_t            total_cycles;
    uint64_t            active_cycles;
    uint64_t            stall_cycles;     // 因资源 stall 的 cycle
    
    // 追踪队列 (用于 Perfetto)
    std::vector<ScheduleEvent> schedule_trace;
};
```

### 4.2 Runtime Hook 设计

#### 4.2.1 Hook 架构

```cpp
// Runtime Hook - 通过 LD_PRELOAD 拦截 ROCm API
class RuntimeHook {
public:
    // 初始化 hook
    static void initialize();
    
    // 拦截的 API
    static hipError_t hipModuleLaunchKernel(
        hipFunction_t f,
        uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        uint32_t sharedMem,
        hipStream_t stream,
        void** args
    );
    
    static hipError_t hipMemcpy(void* dst, const void* src,
                                 size_t size, hipMemcpyKind kind);
    
    // 获取当前执行上下文
    static ExecutionContext& get_context();
    
private:
    static bool             initialized_;
    static ExecutionContext context_;
};
```

#### 4.2.2 执行上下文

```cpp
// 执行上下文 - 管理整个 GPU 程序的执行
class ExecutionContext {
public:
    // 内核启动记录
    struct KernelLaunch {
        std::string     name;
        uint32_t        grid_x, grid_y, grid_z;
        uint32_t        block_x, block_y, block_z;
        uint32_t        shared_mem;
        void**          args;
        uint64_t        host_timestamp;
    };
    
    // 内存传输记录
    struct MemTransfer {
        void*           dst;
        const void*     src;
        size_t          size;
        hipMemcpyKind   kind;
    };
    
    // 添加记录
    void record_kernel_launch(const KernelLaunch& launch);
    void record_mem_transfer(const MemTransfer& transfer);
    
    // 验证结果
    bool verify_results();
    
private:
    std::vector<KernelLaunch>   kernel_launches_;
    std::vector<MemTransfer>    mem_transfers_;
    std::vector<uint8_t>        golden_output_;  // 参考输出
};
```

### 4.3 解码器设计

#### 4.3.1 AMDGCN 指令格式

```cpp
// AMD GPU 指令集架构版本
enum class ISAVersion : uint8_t {
    GCN_1_0,   // Hawaii, Tonga
    GCN_1_1,   // Fiji, Polaris
    GCN_1_2,   // Vega
    GCN_1_4,   // Raven Ridge
    RDNA_1,    // Navi
    RDNA_2,    // Big Navi, CDNA
    CDNA_2     // CDNA2 (MI200)
};

// 指令格式解析
struct InstructionFormat {
    uint8_t  opcode_bits;    // 操作码位数
    uint8_t  opcode_offset;  // 操作码偏移
    uint8_t  dst_bits;       // 目的操作数位数
    uint8_t  src0_bits;      // 源操作数0位数
    uint8_t  src1_bits;      // 源操作数1位数
    uint8_t  src2_bits;      // 源操作数2位数
    bool     has_vop3;       // 是否有 VOP3 扩展
    bool     has_mimg;       // 是否有 MIMG 扩展
};

// AMDGCN 指令定义数据库
class InstrDB {
public:
    static const Instruction* get_instr(uint32_t encoding, ISAVersion version);
    static uint32_t get_latency(const Instruction* instr, const CU& cu);
    
private:
    static std::unordered_map<uint32_t, Instruction> instr_table_;
};
```

### 4.4 Cycle 模拟引擎

#### 4.4.1 模拟循环

```cpp
// Cycle 模拟器核心
class CycleSimulator {
public:
    // 初始化
    void initialize(const std::vector<CUConfig>& cu_configs);
    
    // 添加工作负载
    void add_wave(std::shared_ptr<Wave> wave);
    
    // 运行模拟
    void run(uint64_t max_cycles = UINT64_MAX);
    
    // 获取结果
    CycleStats get_stats() const;
    
private:
    // 单个 cycle 的模拟
    void simulate_cycle();
    
    // Wave 调度决策
    void schedule_waves();
    
    // 执行单元分配
    bool allocate_exec_unit(std::shared_ptr<Wave> wave, ExecUnit unit);
    
    // 资源冲突检测
    bool check_resource_conflict(const Wave& wave, const Instruction& instr);
    
    // 数据结构
    std::vector<ComputeUnit> compute_units_;
    std::priority_queue<WaveEvent> wave_events_;  // 按 cycle 排序的事件
    
    uint64_t current_cycle_;
    CycleStats stats_;
};

// Wave 调度事件
struct WaveEvent {
    uint64_t            cycle;
    EventType           type;
    uint32_t            cu_id;
    uint32_t            wave_id;
    
    bool operator<(const WaveEvent& other) const {
        return cycle > other.cycle;  // 用于 min-heap
    }
};

enum class EventType {
    WAVE_DISPATCH,      // Wave 分发
    WAVE_YIELD,         // Wave 让出
    WAVE_RESUME,        // Wave 恢复
    WAVE_COMPLETE,      // Wave 完成
    INSTR_COMPLETE,     // 指令完成
    MEMORY_RESPONSE     // 内存响应
};
```

#### 4.4.2 执行单元模型

```cpp
// 执行单元延迟表
class LatencyTable {
public:
    // 获取指令延迟
    static uint32_t get_latency(const Instruction& instr, ExecUnit unit) {
        return latency_table_[instr.category][unit];
    }
    
    // 获取指令吞吐量
    static uint32_t get_throughput(const Instruction& instr, ExecUnit unit) {
        return throughput_table_[instr.category][unit];
    }
    
private:
    // 延迟表 (cycle)
    static constexpr uint32_t latency_table_[InstrCategory::MAX][ExecUnit::MAX] = {
        // VALU, SALU, VMEM, SMEM, LDS, GDS, BRANCH
        {4,  0,   0,    0,    0,   0,   0},    // VALU_SIMPLE (VADD, VMUL)
        {16, 0,   0,    0,    0,   0,   0},    // VALU_COMPLEX (VEXP, VLOG)
        {0,  4,   0,    0,    0,   0,   0},    // SALU_SIMPLE (SADD)
        {0,  4,   0,    0,    0,   0,   0},    // SALU_COMPLEX
        {0,  0,   4,    0,    0,   0,   0},    // VMEM_LOAD
        {0,  0,   4,    0,    0,   0,   0},    // VMEM_STORE
        {0,  0,   0,    4,    0,   0,   0},    // SMEM_LOAD
        {0, 0,   0,    4,    0,   0,   0},    // SMEM_STORE
        {0,  0,   0,    0,   4,   0,   0},    // LDS_LOAD
        {0,  0,   0,    0,   4,   0,   0},    // LDS_STORE
        {4,  0,   0,    0,    0,   0,   4},    // BRANCH
    };
    
    // 吞吐量表 (cycle)
    static constexpr uint32_t throughput_table_[InstrCategory::MAX][ExecUnit::MAX] = {
        // VALU, SALU, VMEM, SMEM, LDS, GDS, BRANCH
        {1,  0,   0,    0,    0,   0,   0},    // VALU_SIMPLE
        {4,  0,   0,    0,    0,   0,   0},    // VALU_COMPLEX
        {0,  1,   0,    0,    0,   0,   0},    // SALU_SIMPLE
        {0,  4,   0,    0,    0,   0,   0},    // SALU_COMPLEX
        {0,  0,   4,    0,    0,   0,   0},    // VMEM_LOAD (需要 4 cycle)
        {0,  0,   4,    0,    0,   0,   0},    // VMEM_STORE
        {0,  0,   0,    4,    0,   0,   0},    // SMEM_LOAD
        {0,  0,   0,    4,    0,   0,   0},    // SMEM_STORE
        {0,  0,   0,    0,   4,   0,   0},    // LDS_LOAD
        {0,  0,   0,    0,   4,   0,   0},    // LDS_STORE
        {1,  0,   0,    0,    0,   0,   1},    // BRANCH
    };
};
```

### 4.5 内存系统模型

#### 4.5.1 内存层次

```cpp
// 内存层次模型
struct MemoryHierarchy {
    // L1 Data Cache
    struct L1Cache {
        uint32_t    size_kb;           // 16 KB
        uint32_t    line_size;         // 64 bytes
        uint32_t    associativity;     // 4-way
        uint32_t    latency;           // ~4 cycles
        uint32_t    bandwidth;         // 32 bytes/cycle
    };
    
    // L2 Cache
    struct L2Cache {
        uint32_t    size_mb;           // 8 MB (shared)
        uint32_t    line_size;         // 64 bytes
        uint32_t    associativity;     // 16-way
        uint32_t    latency;           // ~40 cycles
        uint32_t    bandwidth;         // 64 bytes/cycle
    };
    
    // HBM/Device Memory
    struct DeviceMemory {
        uint64_t    size_gb;           // 16 GB
        uint32_t    latency;           // ~400 cycles
        uint32_t    bandwidth_gbps;    // 2 TB/s
    };
    
    // LDS (Local Data Share)
    struct LDS {
        uint32_t    size_kb;           // 64 KB per CU
        uint32_t    latency;           // ~32 cycles (bank conflict considered)
    };
    
    L1Cache       l1;
    L2Cache       l2;
    DeviceMemory  device;
    LDS           lds;
};

// 内存访问追踪
struct MemoryAccess {
    uint64_t    cycle;
    uint64_t    addr;
    uint32_t    size;
    AccessType  type;       // READ, WRITE
    MemoryLevel level;      // REG, LDS, L1, L2, DEVICE
    uint32_t    latency;    // 实际延迟
    bool        hit;        // 是否命中缓存
};
```

---

## 五、Wave 调度策略

### 5.1 调度状态机

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Wave 调度状态机                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    ┌─────────┐                                                                    │
│    │  READY  │                                                                    │
│    └────┬────┘                                                                    │
│         │ dispatch                                                                 │
│         ▼                                                                        │
│    ┌─────────┐  decode   ┌─────────┐  execute  ┌──────────┐  complete ┌────────┐│
│    │DISPATCH │ ───────▶  │DECODING │ ───────▶  │EXECUTING │ ────────▶│COMPLETE││
│    └─────────┘           └─────────┘           └─────┬────┘           └────────┘│
│                                                      │                             │
│                              ┌────────────────────────┤                             │
│                              │                        │                             │
│                              ▼                        ▼                             │
│                       ┌─────────────┐           ┌─────────────┐                   │
│                       │   YIELDED   │           │   WAITING   │                   │
│                       │(s_sleep等)  │           │(资源冲突/    │                   │
│                       └──────┬──────┘           │ 同步等待)    │                   │
│                              │                  └──────┬──────┘                   │
│                              │                         │                           │
│                              │    resource_available   │                           │
│                              └─────────────────────────┘                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 调度算法

```cpp
// Wave 调度器
class WaveScheduler {
public:
    // 调度决策
    ScheduleDecision make_decision(
        const std::vector<std::shared_ptr<Wave>>& ready_waves,
        const ComputeUnit& cu
    );
    
private:
    // 评分函数 - 决定哪个 wave 应该被调度
    float score_wave(const Wave& wave, const ComputeUnit& cu) {
        float score = 0.0f;
        
        // 优先级因素
        score += wave.priority * 10.0f;
        
        // 资源利用因素
        if (wave.occupied_units.empty()) {
            score += 5.0f;  // 新 wave 优先
        }
        
        // 缓存亲和性
        if (wave.last_cu_id == cu.cu_id) {
            score += 3.0f;  // 保持在同一 CU
        }
        
        // 饥饿避免
        score += wave.wait_cycles * 0.1f;
        
        return score;
    }
    
    // 调度策略
    enum class Policy {
        ROUND_ROBIN,      // 轮转
        SCORE_BASED,      // 基于评分
        FIFO,             // 先进先出
        LATEST_FIRST      // 优先调度新 wave
    };
    
    Policy policy_ = Policy::SCORE_BASED;
};
```

### 5.3 气泡检测与统计

```cpp
// 性能瓶颈分析
class BottleneckAnalyzer {
public:
    // 瓶颈类型
    enum class BottleneckType : uint8_t {
        NONE,           // 无瓶颈
        COMPUTE_BOUND,  // 计算受限
        MEMORY_BOUND,   // 内存受限
        LDS_BOUND,      // LDS 受限
        WAVE_SLOT,      // Wave 槽位不足
        EXEC_UNIT,      // 执行单元不足
        BARRIER,        // 同步屏障
        BRANCH_DIVERGE  // 分支分歧
    };
    
    // 分析结果
    struct AnalysisResult {
        BottleneckType  primary_bottleneck;
        uint64_t        total_cycles;
        uint64_t        active_cycles;
        uint64_t        stall_cycles;
        float           utilization;
        
        std::map<BottleneckType, uint64_t> bottleneck_cycles;
        std::vector<std::string> recommendations;
    };
    
    AnalysisResult analyze(const CycleStats& stats);
    
private:
    // 检测具体瓶颈
    void detect_compute_bound(const CycleStats& stats);
    void detect_memory_bound(const CycleStats& stats);
    void detect_resource_bound(const CycleStats& stats);
};
```

---

## 六、Perfetto 集成

### 6.1 Trace 数据模型

```cpp
// Perfetto 追踪事件
struct PerfettoEvent {
    std::string     name;
    uint64_t        timestamp_ns;   // 纳秒
    Duration        duration_ns;
    Category        category;
    Pid             pid;
    Tid             tid;
    
    // 自定义数据
    nlohmann::json args;
};

// 类别定义
enum class Category {
    KERNEL,        // 内核执行
    WAVE,          // Wave 调度
    INSTR,         // 指令执行
    MEMORY,        // 内存访问
    STALL,         // 气泡/停顿
    CU             // CU 状态
};
```

### 6.2 追踪生成

```cpp
// Perfetto 追踪生成器
class PerfettoTracer {
public:
    // 初始化追踪
    void initialize(const std::string& output_path);
    
    // 记录事件
    void trace_event(const PerfettoEvent& event);
    void trace_duration(const std::string& name, 
                       uint64_t start_ns, 
                       uint64_t end_ns,
                       Category cat);
    
    // 记录异步事件
    void trace_async(const std::string& name,
                     uint64_t timestamp_ns,
                     int64_t id,
                     Category cat);
    
    // 写入追踪文件
    void flush();
    
private:
    // 转换为 Perfetto 格式
    proto::Trace convert_to_proto();
    
    std::string output_path_;
    std::vector<PerfettoEvent> events_;
    uint64_t base_timestamp_;
};
```

### 6.3 可视化时间线

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Perfetto Timeline View                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  CU 0                                                                           
│  ├─ Wave 0   ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  ├─ Wave 1   ░░░████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░│
│  ├─ Wave 2   ░░░░░░░░░████████████████████████████████████████░░░░░░░░░░░░░░░░░│
│  └─ Wave 3   ░░░░░░░░░░░░░░░░░░░░░░░████████████████████████░░░░░░░░░░░░░░░░░░│
│                                                                                 │
│  CU 1                                                                           
│  ├─ Wave 4   ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  ├─ Wave 5   ░░░████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░│
│  └─ Wave 6   ░░░░░░░░░░░░░░░░████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░│
│                                                                                 │
│  ────────┬───────────────┬───────────────┬───────────────┬───────────────      │
│      0   │    1000       │    2000       │    3000       │    4000 (cycles)    │
│                                                                                 │
│  Legend: █ Active  ░ Stall/Waiting                                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 七、输出报告

### 7.1 Cycle 统计报告

```json
{
  "simulation_summary": {
    "total_cycles": 1234567,
    "total_waves": 256,
    "total_instructions": 9876543,
    "wave_utilization": 0.78,
    "cu_utilization": {
      "cu_0": 0.85,
      "cu_1": 0.82,
      "cu_2": 0.79,
      "cu_3": 0.81
    }
  },
  "bottleneck_analysis": {
    "primary": "MEMORY_BOUND",
    "secondary": "WAVE_SLOT",
    "memory_stall_cycles": 234567,
    "compute_cycles": 876543,
    "memory_bound_percentage": 21.3
  },
  "wave_schedule": {
    "avg_wait_cycles": 45,
    "max_wait_cycles": 1234,
    "avg_execution_cycles": 4823,
    "wave_switch_overhead": 12
  },
  "instruction_statistics": {
    "valu_instructions": 5432109,
    "salu_instructions": 1234567,
    "vmem_instructions": 2345678,
    "lds_instructions": 543210,
    "branch_instructions": 321876
  },
  "memory_statistics": {
    "global_memory_accesses": 1234567,
    "local_memory_accesses": 543210,
    "l1_cache_hits": 876543,
    "l1_cache_misses": 234567,
    "l2_cache_hits": 156789,
    "l2_cache_misses": 77778,
    "avg_memory_latency": 234.5
  }
}
```

### 7.2 瓶颈分析报告

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Bottleneck Analysis Report                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Primary Bottleneck: MEMORY_BOUND (21.3% of total cycles)                      │
│  ───────────────────────────────────────────────────────────────────────────   │
│                                                                                 │
│  Memory Bound Analysis:                                                         │
│  • Global memory load stalls: 156,234 cycles                                   │
│  • Global memory store stalls: 78,333 cycles                                   │
│  • L1 cache miss rate: 21.1%                                                   │
│  • L2 cache miss rate: 33.2%                                                   │
│  • Average memory access latency: 234.5 cycles                                 │
│                                                                                 │
│  Recommendations:                                                               │
│  1. Increase data locality - reuse data in registers/LDS                      │
│  2. Use vectorized memory instructions for better bandwidth                    │
│  3. Consider prefetching for predictable access patterns                       │
│  4. Optimize memory coalescing for global memory accesses                     │
│                                                                                 │
│  Secondary Bottleneck: WAVE_SLOT (8.2% of total cycles)                        │
│  ───────────────────────────────────────────────────────────────────────────   │
│                                                                                 │
│  Wave Slot Analysis:                                                            │
│  • CU 0: 8/8 waveslots used, avg wait: 23 cycles                               │
│  • CU 1: 8/8 waveslots used, avg wait: 45 cycles                               │
│  • CU 2: 8/8 waveslots used, avg wait: 67 cycles                               │
│                                                                                 │
│  Recommendations:                                                               │
│  1. Reduce register pressure to enable more waves per CU                       │
│  2. Consider reducing block size to increase wave count                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 八、实现路线图

### 8.1 阶段划分

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Implementation Roadmap                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Phase 1: Core Infrastructure (6 weeks)                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • ELF Parser for AMDGPU .out files                                      │   │
│  │ • Runtime Hook (LD_PRELOAD) for hipModuleLaunchKernel                  │   │
│  │ • Basic Instruction Decoder (GCN 1.0 - GCN 1.2)                        │   │
│  │ • Wave creation from grid/block dimensions                              │   │
│  │ • Basic cycle counter                                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Phase 2: Cycle Simulation Engine (8 weeks)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Instruction execution model (VALU, SALU, VMEM, LDS)                   │   │
│  │ • Basic latency model                                                   │   │
│  │ • Wave scheduling (round-robin)                                        │   │
│  │ • Simple memory hierarchy model                                         │   │
│  │ • Basic result verification                                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Phase 3: Performance Analysis (6 weeks)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Bottleneck detection algorithms                                       │   │
│  │ • Detailed statistics collection                                        │   │
│  │ • Report generation                                                     │   │
│  │ • Optimization suggestions                                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Phase 4: Visualization & Integration (4 weeks)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Perfetto integration                                                  │   │
│  │ • Timeline visualization                                               │   │
│  │ • HTML report generation                                               │   │
│  │ • CI/CD integration                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Phase 5: Advanced Features (Ongoing)                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • RDNA/CDNA support                                                    │   │
│  │ • More accurate timing models                                          │   │
│  │ • Multi-CU simulation                                                  │   │
│  │ • Integration with compiler toolchains                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 验证策略

```cpp
// 验证策略
class VerificationStrategy {
public:
    // 功能正确性验证
    static bool verify_functional(
        const std::vector<uint8_t>& simulator_output,
        const std::vector<uint8_t>& golden_output
    ) {
        // 逐字节比较
        return simulator_output == golden_output;
    }
    
    // 性能合理性验证
    static bool verify_performance_reasonableness(
        const CycleStats& stats,
        const KernelLaunch& launch
    ) {
        // 理论下限检查
        uint64_t theoretical_min = estimate_theoretical_min(launch);
        
        // 实际 cycle 应该在理论下限的 1.1x - 10x 之间
        float ratio = (float)stats.total_cycles / theoretical_min;
        
        return ratio >= 1.1f && ratio <= 10.0f;
    }
    
private:
    static uint64_t estimate_theoretical_min(const KernelLaunch& launch);
};
```

---

## 九、技术规格总结

### 9.1 系统参数

| 参数 | 默认值 | 可配置范围 |
|------|--------|------------|
| 模拟 CU 数量 | 4 | 1-64 |
| 每 CU Wave 槽位 | 8 | 1-16 |
| Wave 大小 | 64 | 32/64 |
| L1 缓存大小 | 16 KB | 8-64 KB |
| L2 缓存大小 | 8 MB | 1-16 MB |
| LDS 大小 | 64 KB | 16-128 KB |
| 全局内存延迟 | 400 cycles | 100-1000 cycles |

### 9.2 输出产物

1. **Cycle 统计报告** (`cycle_report.json`)
2. **Perfetto 追踪文件** (`trace.perfetto`)
3. **HTML 可视化报告** (`report.html`)
4. **Wave 调度日志** (`schedule.log`)

### 9.3 精度目标

- **功能正确性**: 100% (与真实 GPU 结果一致)
- **Cycle 精度**: ±15% (相对于真实硬件)
- **性能趋势**: 准确反映优化效果
- **瓶颈识别**: 准确识别 Top 3 瓶颈

---

本设计方案提供了一个完整的轻量级 AMD GPU cycle 模拟器框架，能够在可接受的性能开销下提供有价值的性能分析数据。通过模块化的设计，系统具有良好的可扩展性，可以根据实际需求添加更多的架构特性和优化。
