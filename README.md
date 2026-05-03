# What is this ?

This is a simple PoC AI agent for optimizing C programs via LLMs.

Some design decisions:
  - not using Claude Tools API because of portability concerns

# How to run

You can run it like
```
$ export GEM5_PATH=$HOME/src/gem5
$ ./optimizer_agent.py -m sonnet --max-trials 10 --tmp-dir ./tmp kernels/dot.c
```

File name can be `-` or even omitted for stdin input.

Expect ~$2 for optimization of matmul with 10 trials on Sonnet
(Haiku is too simple for this task and I have not tried Opus).

# Running in Docker

You can build container via
```
docker build -t optimizer_agent .
```
and the use it:
```
docker run --rm -i \
  -e ANTHROPIC_AUTH_TOKEN=... \
  optimizer_agent -m sonnet --max-trials 10 -v < kernels/dot.c
```
