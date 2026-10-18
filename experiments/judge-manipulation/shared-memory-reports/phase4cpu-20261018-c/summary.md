# Phase 4 shared-memory probe — phase4cpu-20261018-c

The registered run ended **failed_incomplete** at kernel_correctness after 74.256679 seconds of its 2,700-second limit. It stopped with kernel_correctness_failure.

The CUDA correctness process initialized the RTX 4060 Ti, then Triton could not find a C compiler: RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl. The kernel check failed after 51.946351 seconds. Training, checkpoint save, and fresh reload did not run.

The resource monitor recorded 55 resource polls, 55 missed polls, a maximum consecutive-missing streak of 2, and zero resource-limit violations. Windows sampling returned 54 valid samples and one invalid performance-counter sample. Cleanup reaped the kernel process group and stopped the Windows sampler (exit code 0); a post-run process query found no Phase 4 model process.

report.json SHA-256: 80fcee6f348f83a1ecc1f5fb2923c642a4fc83726751d7c581fbf83dff14cb91  
verdict.json SHA-256: c23399a1749c4ce3d41a74a55c5677075716b3c0a96b0c50f71c4038e7305b59  
retention.json confirms identical report hashes in the Linux runtime, retained run, and report directory.
