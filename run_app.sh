#!/bin/bash

echo "🚀 Electronic Nose Coffee - KR260 Launcher"

export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export GDK_BACKEND=x11
export EGL_PLATFORM=x11

xhost +local: 2>/dev/null

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

export OMP_NUM_THREADS=3
export POLARS_MAX_THREADS=2
export RUST_BACKTRACE=1
export RUST_LOG=info

echo "🎨 Starting GUI Application..."
echo "   Rendering: Software (llvmpipe)"

sudo nice -n -10 ./target/release/main_gui

echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
