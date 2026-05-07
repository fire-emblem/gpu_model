#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Detect build directory
if [ -n "${GPU_MODEL_BUILD_DIR:-}" ]; then
  BUILD_DIR="$GPU_MODEL_BUILD_DIR"
elif [ -d "$PROJECT_DIR/build-ninja" ]; then
  BUILD_DIR="$PROJECT_DIR/build-ninja"
else
  BUILD_DIR="$PROJECT_DIR/build"
fi

DEMO="$BUILD_DIR/occupancy_analysis_demo"
if [ ! -f "$DEMO" ]; then
  echo "Demo binary not found at $DEMO"
  echo "Build the project first: cmake --build $BUILD_DIR"
  exit 1
fi

echo "========================================="
echo " Occupancy Analysis Demo (built-in mac500)"
echo "========================================="
"$DEMO" --arch mac500

echo ""
echo "========================================="
echo " Occupancy Analysis Demo (from JSON config)"
echo "========================================="
if [ -f "$PROJECT_DIR/configs/gpu_arch/mac500.json" ]; then
  "$DEMO" --config "$PROJECT_DIR/configs/gpu_arch/mac500.json"
fi

echo ""
echo "========================================="
echo " Occupancy Analysis Demo (gfx90a config)"
echo "========================================="
if [ -f "$PROJECT_DIR/configs/gpu_arch/gfx90a.json" ]; then
  "$DEMO" --config "$PROJECT_DIR/configs/gpu_arch/gfx90a.json"
fi
