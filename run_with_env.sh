#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
export JULIA_PKG_PRESERVE_TIERED_INSTALLED=true
export JULIA_CPU_TARGET="generic;
ivybridge,-xsaveopt,clone_all;
sandybridge,-xsaveopt,clone_all;
haswell,-rdrnd,base(1);
znver1,-clzero,base(1);
znver3,-vaes,-clzero,base(1)"
export JULIA_NUM_PRECOMPILE_TASKS=16

julia --project=@. --threads=auto -i
