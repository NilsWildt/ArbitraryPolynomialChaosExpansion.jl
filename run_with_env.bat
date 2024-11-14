@echo off
set CUDA_VISIBLE_DEVICES=
set JULIA_PKG_PRESERVE_TIERED_INSTALLED=true
set JULIA_CPU_TARGET=generic;ivybridge,-xsaveopt,clone_all;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);znver1,-clzero,base(1);znver3,-vaes,-clzero,base(1)
set JULIA_NUM_THREADS=auto
set JULIA_NUM_PRECOMPILE_TASKS=16

julia --project=@. -i --threads=auto -e "using Pkg; Pkg.instantiate(); using CpuId; println(cpu_target_string())"