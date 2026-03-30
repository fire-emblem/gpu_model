> [!NOTE]
> 外部参考/对比文档。用于记录其他方案与比较分析，不直接定义当前仓库实现。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。

AMD GPU 轻量级 Cycle Simulator 设计文档
1. 项目目标
1.1 核心目标

构建一个面向 AMD GPU（GCN / RDNA）的轻量级 cycle simulator，用于：

编译器优化（instruction scheduling / register allocation）
算子库优化（tiling / memory access / vectorization）

输入：

ISA trace（已调度）

输出：

Kernel 总执行周期（total cycles）
Stall breakdown（瓶颈分析）
Resource utilization（资源利用率）
Timeline（用于可视化）
1.2 非目标（明确范围）

不实现：

RTL级精确模拟
Pipeline逐级建模（IF/ID/EX等）
Cache一致性协议
精确bank conflict模拟
指令cache / fetch模型
1.3 成功标准

模拟器应能回答：

当前kernel性能瓶颈在哪里？
修改调度是否提升性能？
是否存在latency未被隐藏？
occupancy是否成为限制因素？
2. 总体架构

模块划分如下：

ISA Trace Loader
Kernel Simulator
CU Model
Wave Scheduler
Scoreboard
Memory Model
Metrics & Profiler
Trace Exporter（Perfetto）

数据流：

ISA Trace → Simulator → Metrics → Trace输出

3. 指令模型（ISA抽象）
3.1 指令类型

定义统一抽象类别：

VALU（向量ALU）
SALU（标量ALU）
VMEM（全局内存访问）
LDS（共享内存）
BRANCH（分支）
BARRIER（同步）
3.2 指令结构

字段包括：

类型（type）
延迟（latency）
源寄存器列表（src_regs）
目标寄存器列表（dst_regs）
内存访问字节数（mem_bytes，可选）
4. Wave模型
4.1 Wave状态

Wave在模拟器中是基本执行单元，状态包括：

READY（可执行）
RUNNING（当前被调度）
STALLED_DEP（依赖阻塞）
STALLED_MEM（内存等待）
FINISHED（完成）
4.2 Wave结构

Wave包含：

id
pc（程序计数器）
ready_cycle（下一次可执行时间）
状态（status）
scoreboard（依赖跟踪）
outstanding_mem（未完成内存请求数）
5. Scoreboard设计（依赖模型）

用于追踪寄存器依赖（RAW hazard）。

维护：

reg → ready_cycle 映射

逻辑：

如果某条指令的源寄存器未ready，则stall
执行指令后，更新目标寄存器ready时间
6. CU（Compute Unit）模型

CU负责调度多个wave。

包含资源：

issue width（每cycle最大发射数）
VALU slots
SALU slots
Memory slots
wave列表
7. 核心执行模型（Cycle推进）
7.1 主循环

每个cycle执行：

cycle++
找到所有ready wave
如果没有ready wave → idle cycle++
否则尝试发射指令
更新wave状态
检查结束条件

结束条件：

所有wave状态为FINISHED
7.2 调度策略（初版）
Round-robin
或简单遍历

后续可扩展：

priority-based
latency-aware
7.3 Issue逻辑

对于每个ready wave：

取当前指令
检查scoreboard依赖
不满足 → dependency stall
检查资源是否可用
不满足 → issue stall
否则执行
7.4 执行行为

根据指令类型：

VALU / SALU：
更新scoreboard
下一cycle可继续
VMEM：
设置 wave 为 STALLED_MEM
ready_cycle = 当前cycle + memory latency
LDS：
类似VMEM，但延迟较低
8. Memory Model设计
8.1 目标

提供：

latency估计
memory stall建模
8.2 简化模型（推荐）

latency = base_latency + miss_rate × penalty

参数：

base_latency（如200 cycles）
miss_rate（估计）
penalty（如300 cycles）
8.3 可扩展
coalescing分析
stride检测
bandwidth限制（queue模型）
9. Occupancy模型

计算最大可并发wave数：

occupancy = min(
reg_limit / reg_per_wave,
lds_limit / lds_per_wave,
max_wave_slots
)

影响：

latency hiding能力
idle概率
10. 性能指标设计
10.1 核心指标
total_cycles
idle_cycles（无wave可执行）
dependency_stall
memory_stall
issue_stall
valu_used_cycles
memory_used_cycles
10.2 派生指标
IPC = total_instructions / total_cycles
VALU utilization = valu_used / total_cycles
idle ratio = idle_cycles / total_cycles
10.3 Stall分类

每个cycle归因：

NO_WAVE（frontend空泡）
DEPENDENCY
MEMORY
ISSUE_LIMIT
11. 空泡（Bubble）建模
11.1 Frontend空泡

条件：

没有任何wave ready

表现：

idle cycle++
11.2 Backend空泡

条件：

有wave ready，但资源未打满

表现：

issue slots未满
11.3 资源利用率

每cycle统计：

used_valu_slots
used_mem_slots

计算：

utilization = used / capacity
12. Timeline可视化设计

使用工具：Perfetto

12.1 映射关系
CU → process
Wave → thread
指令 → slice
利用率 → counter
12.2 Slice（时间片）

表示指令执行或stall：

字段：

name（如 VALU / MEM_STALL）
ts（开始时间）
dur（持续时间）
pid（CU）
tid（wave id）
12.3 Counter（统计曲线）

用于展示：

ready_waves数量
VALU利用率
memory利用率
12.4 示例结构

时间轴上可观察：

Wave执行与等待
CU是否idle
stall分布
13. 状态机模型

Wave状态流转：

READY → ISSUE → 执行

执行后：

若依赖 → STALLED_DEP
若内存 → STALLED_MEM
若完成 → FINISHED

STALLED_MEM → 到达ready_cycle → READY

14. 最小C++骨架（可直接实现）
14.1 Instruction

struct Instruction {
int type;
int latency;
std::vector<int> src;
std::vector<int> dst;
};

14.2 Scoreboard

struct Scoreboard {
std::unordered_map<int, int> reg_ready;

bool ready(const Instruction& inst, int cycle) {
    for (auto r : inst.src) {
        if (reg_ready[r] > cycle) return false;
    }
    return true;
}

void update(const Instruction& inst, int cycle) {
    for (auto r : inst.dst) {
        reg_ready[r] = cycle + inst.latency;
    }
}

};

14.3 Wave

struct Wave {
int pc;
int ready_cycle;
int status;
Scoreboard sb;
};

14.4 Simulator主循环

while (!finished) {
cycle++;

ready_waves = find_ready_waves();

if (ready_waves.empty()) {
    idle_cycles++;
    continue;
}

for (wave in ready_waves) {
    inst = program[wave.pc];

    if (!wave.sb.ready(inst, cycle)) {
        dep_stall++;
        continue;
    }

    execute(wave, inst);
}

}

14.5 执行函数

void execute(Wave& w, Instruction& inst) {
w.sb.update(inst, cycle);

if (inst.type == VMEM) {
    w.ready_cycle = cycle + mem_latency;
}

w.pc++;

}

15. 输出结果（最终形态）

输出内容应包括：

Kernel总周期（最重要）
Stall breakdown
Resource utilization
Occupancy信息
Timeline（Perfetto trace）
16. 后续扩展方向
更真实调度器（dual issue / priority）
Memory带宽模型
自动调优（autotuning loop）
LLVM集成（MIR输入）
ML cost model
17. 核心设计原则总结

一句话总结架构：

Wave级事件驱动模拟 + Scoreboard依赖跟踪 + 统计内存模型 + Timeline可视化
