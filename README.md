# What is this ?

This is a simple PoC AI agent for optimizing C programs via LLMs.

You can run it like
```
$ export GEM5_PATH=$HOME/src/gem5
$ ./optimizer_agent.py -m sonnet --max-trials 10 --tmp-dir ./tmp kernels/matmul.c
```

Expect ~$2 for optimization of matmul with 10 trials on Sonnet
(Haiku is too simple and I have not tried Opus).

Some design decisions:
  - not using Claude Tools API because of portability concerns
