# Granite one-shot prefill experiment

This branch sets Granite's mandatory prefill chunk size to `None`, so a
64-token prompt is no longer padded to a 512-token query chunk. A 1,984-token
prompt likewise runs as one prefill. Stick alignment and independently sized
SDPA-compatible KV capacity still apply; this is not arbitrary unaligned input
support.

This is a draft experiment, not a qualified default for all Granite workloads.
The integrated Granite 4.2 8B FP16 TP4 runs covered 64 and 1,984 prompt tokens,
10 generated tokens, and exact output comparisons. Longer prompts, memory
limits, and other Granite variants need qualification before merging.

HF-adapters PR #575 proposes narrower short-prompt compaction within one cache
tile. It is an overlapping alternative for production policy, not a prerequisite
for this experiment. Resolve the policy before merging either implementation;
this branch also supports the 1,984-token one-shot experiment that #575 does not.

The mask optimization and prefill/decode profiler scopes are separate changes.
