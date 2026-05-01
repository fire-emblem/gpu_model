#!/usr/bin/env bash
# =============================================================================
# install_deps.sh - 一键安装 gpu_model 项目本地依赖工具
#
# 依赖内容：
#   - cmake          : pip install -> ~/.local/bin/
#   - ccache        : conda-forge -> tools/conda/
#   - llvm-tools    : conda-forge -> tools/llvm/  (也包含在 tools/hipcc/ 中)
#   - hipcc         : conda-forge -> tools/hipcc/ (需要 ROCm device lib 才能编译 HIP)
#   - mold          : GitHub releases 预编译 (可选) -> tools/mold/
#   - HIP 头文件    : 复制 -> tools/hip/include/
#
# 所有工具安装到项目 tools/ 目录下。
#
# 使用方式：
#   ./scripts/install_deps.sh
#   source scripts/setup_env.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_DIR}/tools"
CONDA_CCACHE="${TOOLS_DIR}/conda"
CONDA_LLVM="${TOOLS_DIR}/llvm"
CONDA_HIPCC="${TOOLS_DIR}/hipcc"
HIP_DIR="${TOOLS_DIR}/hip/include"

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
log_info()  { echo "[install_deps] INFO: $*"; }
log_warn()  { echo "[install_deps] WARN: $*" >&2; }
log_error() { echo "[install_deps] ERROR: $*" >&2; }

check_command() { command -v "$1" &>/dev/null; }

# -----------------------------------------------------------------------------
# 1. cmake (pip package -> ~/.local/bin/)
# -----------------------------------------------------------------------------
install_cmake() {
    log_info "检查 cmake ..."
    local cmake_bin="${HOME}/.local/bin/cmake"
    if [[ -x "${cmake_bin}" ]]; then
        log_info "cmake 已安装: ${cmake_bin} ($("${cmake_bin}" --version | head -1))"
        return 0
    fi
    log_info "安装 cmake (pip install --user cmake) ..."
    pip install --user cmake
    if [[ -x "${cmake_bin}" ]]; then
        log_info "cmake 安装成功"
    else
        log_error "cmake 安装失败"; return 1
    fi
}

# -----------------------------------------------------------------------------
# 2. ccache (conda-forge -> tools/conda/)
# -----------------------------------------------------------------------------
install_ccache() {
    log_info "检查 ccache ..."
    local bin="${CONDA_CCACHE}/bin/ccache"
    if [[ -x "${bin}" ]] && LD_LIBRARY_PATH="${CONDA_CCACHE}/lib:${LD_LIBRARY_PATH:-}" "${bin}" --version &>/dev/null; then
        log_info "ccache 已安装: ${bin}"
        return 0
    fi
    if ! check_command conda; then
        log_error "conda 未找到，请先安装 miniconda3"; return 1
    fi
    log_info "安装 ccache -> ${CONDA_CCACHE} ..."
    conda create -y -p "${CONDA_CCACHE}" ccache
    [[ -x "${bin}" ]] && log_info "ccache 安装成功" || { log_error "ccache 安装失败"; return 1; }
}

# -----------------------------------------------------------------------------
# 3. LLVM 工具链 (conda-forge -> tools/llvm/)
# -----------------------------------------------------------------------------
install_llvm_tools() {
    log_info "检查 LLVM 工具链 ..."
    local tools=("llvm-mc" "llvm-readelf" "llvm-readobj" "llvm-objcopy" "llvm-objdump")
    local all_found=true
    for t in "${tools[@]}"; do
        if [[ ! -x "${CONDA_LLVM}/bin/${t}" ]]; then all_found=false; break; fi
    done
    if $all_found; then
        log_info "LLVM 工具链已安装: ${CONDA_LLVM}"
        return 0
    fi
    # 如果 hipcc conda 已有完整 llvm tools，创建符号链接
    if [[ -x "${CONDA_HIPCC}/bin/llvm-mc" ]]; then
        log_info "发现 hipcc 环境中的 LLVM 工具，创建符号链接 -> ${CONDA_LLVM}"
        mkdir -p "${CONDA_LLVM}/bin"
        for t in "${tools[@]}"; do
            if [[ -x "${CONDA_HIPCC}/bin/${t}" ]] && [[ ! -e "${CONDA_LLVM}/bin/${t}" ]]; then
                ln -sf "${CONDA_HIPCC}/bin/${t}" "${CONDA_LLVM}/bin/${t}"
            fi
        done
        log_info "LLVM 符号链接创建完成"
        return 0
    fi
    if ! check_command conda; then
        log_error "conda 未找到"; return 1
    fi
    log_info "安装 llvm-tools -> ${CONDA_LLVM} ..."
    conda create -y -p "${CONDA_LLVM}" -c conda-forge "llvm-tools=14"
    for t in "${tools[@]}"; do
        local v="${t}-14"
        if [[ -x "${CONDA_LLVM}/bin/${v}" ]] && [[ ! -e "${CONDA_LLVM}/bin/${t}" ]]; then
            ln -s "${v}" "${CONDA_LLVM}/bin/${t}"
        fi
    done
    log_info "LLVM 工具链安装成功"
}

# -----------------------------------------------------------------------------
# 4. hipcc (conda-forge -> tools/hipcc/)
#   注意: hipcc 编译 HIP kernel 还需要 ROCm device library，
#         在非 ROCm 环境中 hipcc 可用但编译可能失败。
# -----------------------------------------------------------------------------
install_hipcc() {
    log_info "检查 hipcc ..."
    local bin="${CONDA_HIPCC}/bin/hipcc"
    if [[ -x "${bin}" ]]; then
        log_info "hipcc 已安装: ${bin}"
        return 0
    fi
    if ! check_command conda; then
        log_warn "conda 未找到，跳过 hipcc 安装"
        return 0
    fi
    log_info "安装 hipcc -> ${CONDA_HIPCC} ..."
    conda create -y -p "${CONDA_HIPCC}" -c conda-forge hipcc
    if [[ -x "${bin}" ]]; then
        log_info "hipcc 安装成功"
        log_warn "hipcc 编译 HIP kernel 需要 ROCm device library，当前环境可能缺失"
    else
        log_warn "hipcc 安装失败"
    fi
}

# -----------------------------------------------------------------------------
# 5. mold (可选, GitHub releases -> tools/mold/)
# -----------------------------------------------------------------------------
install_mold() {
    log_info "检查 mold ..."
    if [[ -x "${TOOLS_DIR}/mold/bin/mold" ]]; then
        log_info "mold 已安装"; return 0
    fi
    local version="2.31.0"
    local url="https://github.com/rui314/mold/releases/download/v${version}/mold-${version}-x86_64-linux.tar.gz"
    local archive="${TOOLS_DIR}/mold.tar.gz"
    log_info "下载 mold ${version} ..."
    if curl -fSL --connect-timeout 30 --max-time 300 "${url}" -o "${archive}" 2>/dev/null; then
        tar -xzf "${archive}" -C "${TOOLS_DIR}"
        rm -f "${archive}"
        log_info "mold 安装成功"
    else
        log_warn "mold 下载失败 (网络问题)，cmake 会回退到系统 ld"
        rm -f "${archive}"
    fi
}

# -----------------------------------------------------------------------------
# 6. HIP 头文件 (-> tools/hip/include/)
# -----------------------------------------------------------------------------
install_hip_headers() {
    log_info "检查 HIP 头文件 ..."
    if [[ -f "${HIP_DIR}/hip_runtime_api.h" ]]; then
        log_info "HIP 头文件已安装: ${HIP_DIR}"; return 0
    fi
    # 从已有来源复制
    local hip_sources=(
        "${HOME}/softmax/miniconda3/envs/maca-torch/lib/python3.10/site-packages/triton/backends/amd/include/hip"
        "/opt/rocm/include/hip"
    )
    for src in "${hip_sources[@]}"; do
        if [[ -f "${src}/hip_runtime_api.h" ]]; then
            log_info "复制 HIP 头文件: ${src} -> ${HIP_DIR}"
            mkdir -p "${HIP_DIR}"
            cp -r "${src}/"* "${HIP_DIR}/"
            return 0
        fi
    done
    log_warn "HIP 头文件未找到，请手动复制到 ${HIP_DIR}/"
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
main() {
    log_info "========================================"
    log_info "gpu_model 本地依赖安装脚本"
    log_info "项目目录: ${PROJECT_DIR}"
    log_info "工具目录: ${TOOLS_DIR}"
    log_info "========================================"

    mkdir -p "${TOOLS_DIR}"

    install_cmake
    install_ccache
    install_llvm_tools
    install_hipcc
    install_mold
    install_hip_headers

    log_info ""
    log_info "========================================"
    log_info "安装完成！"
    log_info "========================================"
    log_info ""
    log_info "  source scripts/setup_env.sh"
    log_info ""
    log_info "工具目录:"
    log_info "  tools/conda/    ccache"
    log_info "  tools/llvm/     LLVM 工具链 (llvm-mc, llvm-readelf 等)"
    log_info "  tools/hipcc/    hipcc (编译 HIP kernel 需 ROCm device lib)"
    log_info "  tools/mold/     mold linker (可选)"
    log_info "  tools/hip/      HIP 头文件"
    log_info ""
}

main "$@"
