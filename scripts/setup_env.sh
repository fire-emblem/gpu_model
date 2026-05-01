#!/usr/bin/env bash
# =============================================================================
# setup_env.sh - 设置 gpu_model 项目本地工具环境变量
#
# 使用方式:
#   source scripts/setup_env.sh
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_DIR}/tools"

# cmake via pip
export PATH="${HOME}/.local/bin:${PATH}"

# ROCm 6.2 完整工具链 (hipcc, clang-18, llvm tools, device libs)
ROCM_INSTALL="${HOME}/tools/rocm/rocm/opt/rocm-6.2.0"
if [[ -d "${ROCM_INSTALL}" ]]; then
    export ROCM_PATH="${ROCM_INSTALL}"
    export HIP_PATH="${ROCM_INSTALL}"
    export HIP_PLATFORM=amd
    export HIP_COMPILER=clang
    export HIP_DEVICE_LIB_PATH="${ROCM_INSTALL}/amdgcn/bitcode"
    export PATH="${ROCM_INSTALL}/bin:${ROCM_INSTALL}/lib/llvm/bin:${PATH}"
    export LD_LIBRARY_PATH="${ROCM_INSTALL}/lib:${ROCM_INSTALL}/lib64:${ROCM_INSTALL}/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
fi

# 项目本地工具: ccache, mold
export PATH="${TOOLS_DIR}/conda/bin:${TOOLS_DIR}/mold-2.31.0-x86_64-linux/bin:${PATH}"

# ccache 依赖库
if [[ -d "${TOOLS_DIR}/conda/lib" ]]; then
    export LD_LIBRARY_PATH="${TOOLS_DIR}/conda/lib:${LD_LIBRARY_PATH:-}"
fi

echo "[setup_env] 环境变量已更新"
echo "  tools: ${TOOLS_DIR}"
echo "  ROCM_PATH: ${ROCM_PATH:-未设置}"
