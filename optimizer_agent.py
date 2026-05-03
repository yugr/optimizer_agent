#!/usr/bin/env python3

# TODO:
#   - unify work with files
#   - support for LLVM IR inputs ?
#   - use typing ?
#   - rewrite explicit verbosity w/ logging
#   - use any-llm instead of vendor API
#   - unify retry pattern in generate_test, generate_bench and _apply_opt
#     if it's not too unreadable (Stratego/XT-like combinators ?)
#   - split to multiple files ?
#   - unittests

import argparse
import copy
import atexit
from dataclasses import dataclass
from heapq import *
import os
from pathlib import Path
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
from typing import NoReturn

from anthropic import Anthropic


# Path to this script (to look for prompts, etc.)
ROOT = Path(__file__).parent


# Path for temp files
TMP = None


# Verbosity
VERBOSE = -1


# Max number of retries in various places
RETRIES = 2  # Just to save tokens


# How many optimizations we try for each version
FANOUT = 2  # Just to save tokens...


# Exploration factor when choosing next candidate
EXPLORE_FACTOR = 0.2


# How many optimizations to try
MAX_TRIALS = 5  # Just to save tokens


# Max time for single benchmark run
BENCH_TIMEOUT = 15


# Has to be rather high even for matmul :(
TOKEN_LIMIT = 10000


def warn(msg):
    """
    Print nicely-formatted warning message.
    """
    me = Path(__file__).name
    sys.stderr.write(f"{me}: warning: {msg}\n")


def error(msg) -> NoReturn:
    """
    Print nicely-formatted error message and exit.
    """
    me = Path(__file__).name
    sys.stderr.write(f"{me}: error: {msg}\n")
    sys.exit(1)


def warn_if(cond, msg):
    if cond:
        warn(msg)


def error_if(cond, msg):
    if cond:
        error(msg)


def run(cmd, fatal=False, tee=False, timeout=None, **kwargs):
    """
    Simple wrapper for subprocess.run.
    """
    if isinstance(cmd, str):
        cmd = re.split(r" +", cmd)
    else:
        cmd = [str(arg) for arg in cmd]
    if VERBOSE > 1:
        print(cmd)
    t1 = time.perf_counter_ns()
    p = subprocess.run(
        cmd,
        timeout=timeout,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    t2 = time.perf_counter_ns()
    out = p.stdout.decode()
    err = p.stderr.decode()
    if fatal and p.returncode != 0:
        cmds = " ".join(cmd)
        error(f"'{cmds}' failed:\n{out}{err}")
    if tee:
        sys.stdout.write(out)
        sys.stderr.write(err)
    return p.returncode, out, err, (t2 - t1) / 1e9


def read_file(filename):
    """Thin wrapper for read."""
    with open(filename, "r") as f:
        return f.read()


def write_file(filename, text):
    """Thin wrapper for write."""
    with open(filename, "w") as f:
        f.write(text)


def make_temp(text, suffix):
    """Make temp file with given text."""
    fd, filename = tempfile.mkstemp(suffix=("_" + suffix), dir=TMP)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return filename


def get_kernel_name(filename):
    """Extract kernel name from compiled binary."""

    # Readelf should be generic enough for all
    _, o, *_ = run(f"readelf -sW {filename}", fatal=True)

    publics = []

    for line in o.split("\n"):
        if not re.search(r"FUNC.*GLOBAL", line):
            continue
        publics.append(re.split(r" +", line)[-1])

    if len(publics) > 1:
        # TODO: ask user ?
        p = ", ".join(publics)
        error(f"too many functions in user file ({p})")
    elif not publics:
        error("no functions in user file")

    return publics[0]


def extract_code(text):
    """Extract code in backticks from prompt."""

    code = None
    in_code = False
    for line in text.split("\n"):
        if line.startswith("```"):
            if in_code:
                break
            else:
                in_code = True
                code = []
        elif in_code:
            code.append(line)

    return "\n".join(code) if code else None


def make_prompt(prompt_name, **kwargs):
    prompt_filename = ROOT / "prompts" / prompt_name
    return string.Template(read_file(prompt_filename)).substitute(**kwargs)


class LLM:
    """Abstract LLM wrappers with common postprocessing."""

    def ask(self, q, ctx=None, code_only=True):
        """Send question within given context."""

        if ctx is None:
            ctx = []  # ctx=[] in arguments is a classical bug

        if VERBOSE > 2:
            print(
                f"""\
Asking LLM:
{q}
Context:
{ctx}
"""
            )

        ans = self._ask_impl(q, ctx)

        if VERBOSE > 2:
            print(
                f"""\
LLM response:
{ans}
"""
            )

        if code_only:
            ans = extract_code(ans)

        return ans

    def _ask_impl(self, q, ctx):
        raise NotImplementedError


class DummyLLM(LLM):
    """
    Dummy LLM which simply returns predefined answers
    (to save tokens in simple tests).
    """

    def __init__(self):
        pass

    def _read_dummy_response(self, filename):
        filename = ROOT / "dummy-responses" / filename
        with open(filename, "r") as f:
            return f.read()

    def _ask_impl(self, q, ctx):
        ctx.append(
            {
                "role": "user",
                "content": q,
            }
        )

        if q == "Hello, Claude":
            ans = self._read_dummy_response("hello.txt")
        elif "Your goal is to write a test function" in q:
            ans = self._read_dummy_response("write_test.txt")
        elif "Your goal is to write function `bench` for benchmarking" in q:
            ans = self._read_dummy_response("write_bench.txt")
        elif "Can you make it faster" in q:
            ans = self._read_dummy_response("update_bench.txt")
        elif "Output optimizations via numbered list" in q:
            ans = self._read_dummy_response("propose_opts.txt")
        elif "Apply only the following optimization" in q:
            # This is silly but we just want to model something here.
            # So we just return same code 50% of time and broken code otherwise.
            code = extract_code(q)
            choice = random.randint(1, 3)
            if choice == 1:  # Return same
                ans = f"""\
```
{code}
```
"""
            elif choice == 2:  # Return broken
                ans = """\
```
void foo(float *a, float *b, float *c, int m, int n, int k) {}
```
"""
            else:  # Report fail
                ans = "NO"
        elif "Your code does not" in q:
            ans = "NO"
        else:
            assert False, "unknown question to dummy model"

        ctx.append(
            {
                "role": "assistant",
                "content": ans,
            }
        )

        return ans


class AnthropicLLM(LLM):
    """Wrapper for Anthropic LLMs."""

    def __init__(self, model):
        self.model = model

        connect_args = {}

        if "ANTHROPIC_AUTH_TOKEN" in os.environ:
            connect_args["auth_token"] = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        elif "ANTHROPIC_API_KEY" in os.environ:
            connect_args["api_key"] = os.environ.get("ANTHROPIC_API_KEY")

        if "ANTHROPIC_BASE_URL" in os.environ:
            connect_args["base_url"] = os.environ.get("ANTHROPIC_BASE_URL")

        self.client = Anthropic(**connect_args)

    def _ask_impl(self, q, ctx):
        # print(client.models.list())

        ctx.append(
            {
                "role": "user",
                "content": q,
            }
        )

        ans = self.client.messages.create(
            max_tokens=TOKEN_LIMIT,
            messages=ctx,
            model=self.model,
        )

        ctx.append(ans)

        text = ""
        for block in ans.content:
            if block.type != "text":
                continue

            text += "\n" + block.text

        return text


class Target:
    """Abstract target platform."""

    def compile(self, file):
        """Compile given file."""
        raise NotImplementedError

    def verify(self, files):
        """Verify that given files may compile, link and run successfully."""
        raise NotImplementedError

    def bench(self, files, timeout):
        """Collect performance for given files."""
        raise NotImplementedError

    def context(self):
        """Additional context for LLM."""
        raise NotImplementedError

    @dataclass
    class CompileError(BaseException):
        "Info about compiler error."
        out: str
        err: str

    @dataclass
    class RuntimeError(BaseException):
        "Info about runtime error."
        out: str
        err: str

    @dataclass
    class PerformanceError(BaseException):
        "Info about performance simulation error."
        out: str
        err: str


class AArch64Target(Target):
    """Target class for AArch64 kernels."""

    def __init__(self, gem5_path):
        self.cc = "clang -target aarch64-linux-gnu"
        # Clang on Ubuntu lacks sanitizer libs for non-x86 platforms
        self.asan_cc = "aarch64-linux-gnu-gcc"
        self.qemu = "qemu-aarch64 -L /usr/aarch64-linux-gnu"
        self.sim = f"{gem5_path}/build/ARM/gem5.opt {gem5_path}/configs/example/arm/starter_se.py"

        self._verify_tools()

    def _verify_tools(self):
        "Verify that tools are installed."

        filename = make_temp("int main() { return 0; }", "dummy.c")

        # Disable Lsan because of issues w/ ptrace, etc.
        os.environ["LSAN_OPTIONS"] = "detect_leaks=0"

        # Regular compile and run

        run(f"{self.cc} -O2 {filename}", fatal=True)
        run(f"{self.qemu} ./a.out", fatal=True)

        run(f"{self.asan_cc} -O2 {filename}", fatal=True)
        run(f"{self.qemu} ./a.out", fatal=True)

        # Sanitized compile and run

        run(f"{self.asan_cc} -O2 -fsanitize=address,undefined {filename}", fatal=True)
        run(f"{self.qemu} ./a.out", fatal=True)

        # Performance simulation

        run(f"{self.cc} -O2 -static {filename}", fatal=True)
        _, o, *_ = run(f"{self.sim} ./a.out", fatal=True)
        assert re.search(
            r"exiting with last active thread context.* @ [0-9]+", o
        ), "unexpected gem5 output"

    def compile(self, file):
        rc, o, e, _ = run(f"{self.cc} -c {file} -o a.out")
        if rc != 0:
            raise Target.CompileError(o, e)

    def verify(self, files):
        for extra_flags in ["", "-fsanitize=address,undefined"]:

            rc, o, e, _ = run(f"{self.asan_cc} -O2 {extra_flags} " + " ".join(files))
            if rc != 0:
                raise Target.CompileError(o, e)

            rc, o, e, _ = run(f"{self.qemu} ./a.out")
            if rc != 0:
                raise Target.RuntimeError(o, e)

        # Also test with clang
        rc, o, e, _ = run(f"{self.cc} -O2 " + " ".join(files))
        if rc != 0:
            raise Target.CompileError(o, e)

    def bench(self, files, timeout):
        rc, o, e, _ = run(f"{self.cc} -O2 -static " + " ".join(files))
        if rc != 0:
            raise Target.CompileError(o, e)

        rc, o, *_ = run(f"{self.sim} ./a.out", timeout=timeout)
        if rc != 0:
            raise Target.RuntimeError(o, e)

        m = re.search(r"exiting with last active thread context.* @ ([0-9]+)", o)
        if m is None:
            raise Target.PerformanceError(o, e)

        return int(m[1])

    def context(self):
        return "You are generating code for AArch64 target."


def generate_test(kernel_name, kernel_code, target, llm):
    """Use LLM to generate testcases for kernel."""

    kernel_filename = make_temp(kernel_code, "kernel.c")

    initial_prompt = make_prompt(
        "write_test.txt",
        KERNEL_NAME=kernel_name,
        KERNEL_CODE=kernel_code,
    )

    ctx = []

    test_code = llm.ask(initial_prompt, ctx)
    test_filename = make_temp(test_code, "test.c")

    driver_filename = make_temp(
        """\
extern void test();

int main() {
    test();
    return 0;
}
""",
        "driver.c",
    )

    def update(what, e):
        prompt = make_prompt(
            "rewrite_test.txt",
            WHAT=what,
            KERNEL_NAME=kernel_name,
            OUTPUT=(e.out + e.err),
        )
        return llm.ask(prompt, ctx)

    for _ in range(RETRIES):
        try:
            target.compile(test_filename)
            target.verify([kernel_filename, test_filename, driver_filename])
            # TODO: check coverage
            return test_code
        except Target.CompileError as e:
            test_code = update("compile", e)
        except Target.RuntimeError as e:
            test_code = update("run successfully", e)

        if test_code is None:
            break

        write_file(test_filename, test_code)

    return None


def generate_bench(kernel_name, kernel_code, target, llm):
    """Use LLM to generate benchmark for kernel."""

    kernel_filename = make_temp(kernel_code, "kernel.c")

    initial_prompt = make_prompt(
        "write_bench.txt",
        KERNEL_NAME=kernel_name,
        KERNEL_CODE=kernel_code,
    )

    ctx = []

    bench_code = llm.ask(initial_prompt, ctx)
    bench_filename = make_temp(bench_code, "bench.c")

    driver_filename = make_temp(
        """\
extern void bench(int n);

int main() {
    bench(1);
    return 0;
}
""",
        "driver.c",
    )

    def update(what, e):
        prompt = make_prompt(
            "rewrite_bench.txt",
            WHAT=what,
            KERNEL_NAME=kernel_name,
            KERNEL_CODE=kernel_code,
            OUTPUT=(e.out + e.err),
        )
        return llm.ask(prompt, ctx)

    def update_timeout():
        return llm.ask(
            "Even for N=1 benchmark takes too long. Can you make it faster?", ctx
        )

    for _ in range(RETRIES):
        try:
            target.compile(bench_filename)
            target.verify([kernel_filename, bench_filename, driver_filename])
            target.bench(
                [kernel_filename, bench_filename, driver_filename],
                timeout=BENCH_TIMEOUT,
            )
            # TODO: calibrate N below so that benched function dominates in program run
            harness_code = f"""\
void bench_N() {{
    bench(1);
}}
"""
            return bench_code, harness_code
        except Target.CompileError as e:
            bench_code = update("compile", e)
        except Target.RuntimeError as e:
            bench_code = update("run successfully", e)
        except subprocess.TimeoutExpired:
            bench_code = update_timeout()

        if bench_code is None:
            break

        write_file(bench_filename, bench_code)

    return None, None


class TreeNode:
    """Single node in LLM-generated search tree."""

    def __init__(
        self,
        kernel_name,
        priority,
        kernel_code,
        bench_code,
        test_filename,
        bench_filename,
        opt=None,
        parent=None,
    ):
        self.kernel_name = kernel_name
        self.priority = priority
        self.kernel_code = kernel_code
        self.bench_code = bench_code
        self.test_filename = test_filename
        self.bench_filename = bench_filename
        self.parent = parent
        self.children = []  # Not expanded initially

        if self.parent is None:
            self.opts = []
            self.depth = 0
        else:
            self.depth = parent.depth + 1
            self.opts = copy.copy(parent.opts)

        if opt is not None:
            self.opts.append(opt)

    def __lt__(self, x):
        return self.priority < x.priority

    def _apply_opt(self, opt, llm, target):
        # TODO: should we inform LLM about old_opts ?
        prompt = make_prompt(
            "apply_opt.txt",
            KERNEL_NAME=self.kernel_name,
            KERNEL_CODE=self.kernel_code,
            TESTCASE_CODE=read_file(self.test_filename),
            # TODO: should we pass benchmark to LLM ? It may cause it to
            # specialize too much...
            BENCH_CODE=self.bench_code,
            OPT=opt,
            TARGET=target.context(),
        )

        ctx = []
        kernel_code = llm.ask(prompt, ctx)
        if kernel_code is None:
            if VERBOSE > 1:
                print("Failed to apply optimization")
            return None, None
        kernel_filename = make_temp(kernel_code, "kernel.c")

        def update(what, e):
            prompt = make_prompt(
                "fix_opt.txt",
                WHAT=what,
                KERNEL_NAME=self.kernel_name,
                KERNEL_CODE=kernel_code,
                OUTPUT=(e.out + e.err),
            )
            return llm.ask(prompt, ctx)

        for _ in range(RETRIES):
            try:
                target.compile(kernel_filename)
                target.verify([kernel_filename, self.test_filename])
                priority = target.bench(
                    [kernel_filename, self.bench_filename], timeout=BENCH_TIMEOUT
                )
                if VERBOSE > 2:
                    print(f"Successfully applied optimization (priority {priority})")
                return kernel_code, priority
            except Target.CompileError as e:
                kernel_code = update("compile", e)
            except Target.RuntimeError as e:
                kernel_code = update("run successfully", e)
            except subprocess.TimeoutExpired:
                # Likely slowdown
                break

            if kernel_code is None:
                break

            write_file(kernel_filename, kernel_code)

        return None, None

    def expand(self, llm, target):
        assert not self.children, "duplicate expand"

        prompt = make_prompt(
            "propose_opts.txt",
            KERNEL_CODE=self.kernel_code,
            FANOUT=FANOUT,
            TARGET=target.context(),
        )

        # TODO:
        #   - add LLVM optimization remarks to prompt
        #   - remember which optimizations have already been tried for this path
        #     (to avoid duplicate applications)
        #   - should I pass bench and testcase here ?
        #   - keeping history may help LLM
        #   - add ultrathink at least for upper levels of tree ?
        ans = llm.ask(prompt, code_only=False)

        opts = []
        for line in ans.split("\n"):
            opt_match = re.match(r"^[0-9]+\.(.*)", line)
            if opt_match:
                opts.append(opt_match[1])

        for i, opt in enumerate(opts):
            if VERBOSE > 1:
                print(f"Trying to apply optimization #{i}...")
            code, priority = self._apply_opt(opt, llm, target)
            if code is not None:
                child = TreeNode(
                    self.kernel_name,
                    priority,
                    code,
                    self.bench_code,
                    self.test_filename,
                    self.bench_filename,
                    opt,
                    self,
                )
                self.children.append(child)


def main_loop(
    kernel_name, kernel_code, bench_code, test_filename, bench_filename, target, llm
):
    """Main agent loop to generate and try optimizations."""

    filename = make_temp(kernel_code, "kernel.c")
    priority = target.bench([filename, bench_filename], BENCH_TIMEOUT)

    root = TreeNode(
        kernel_name, priority, kernel_code, bench_code, test_filename, bench_filename
    )

    q = []
    heappush(q, root)

    best = root
    improvement = 0

    trial = 1

    while q and trial < MAX_TRIALS:
        if VERBOSE > 1:
            print(
                f"Trial #{trial}: {len(q)} items in queue, best improvement +{100 * improvement:.1f}%"
            )
        trial += 1

        # TODO: use MCTS rather than priorities with random exploration
        if random.randint(0, 100) / 100 > EXPLORE_FACTOR:
            reason = "best"
            cand = q.pop()
        else:
            reason = "random"
            k = random.randint(0, len(q) - 1)
            cand = q[k]
            # TODO: how to efficiently remove elements in heapq ?
            del q[k]
            heapify(q)

        if VERBOSE > 1:
            print(
                f"Selecting {reason} candidate (priority {cand.priority}, depth {cand.depth})"
            )

        cand.expand(llm, target)
        if VERBOSE > 1:
            print(f"Candidate expanded to {len(cand.children)} more variants")

        for child in cand.children:
            heappush(q, child)
            if child < best:
                best = child
                improvement = 1 - best.priority / root.priority

    if best == root:
        return None, None

    return best.kernel_code, improvement


def main():
    class Formatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
    ):
        pass

    global EXPLORE_FACTOR
    global FANOUT
    global MAX_TRIALS
    global RETRIES

    parser = argparse.ArgumentParser(
        description="Simple LLM-based program optimizer", formatter_class=Formatter
    )
    parser.add_argument(
        "-e",
        "--explore-factor",
        help="exploration factor when selecting next trial (higher explores more)",
        default=EXPLORE_FACTOR,
        type=float,
    )
    parser.add_argument(
        "-f",
        "--fanout",
        help="how many times to retry LLM codegen on fail",
        default=FANOUT,
        type=int,
    )
    parser.add_argument(
        "--gem5-path",
        help="path to gem5 build (can also use GEM5_PATH environment var)",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model to use (set to 'dummy' for offline testing",
        default="dummy",
    )
    parser.add_argument(
        "-r",
        "--retries",
        help="how many times to retry LLM codegen on fail",
        default=RETRIES,
        type=int,
    )
    parser.add_argument(
        "--quiet",
        "-q",
        help="do not emit any messages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--seed",
        help="seed for random number generator",
        default=None,
    )
    parser.add_argument(
        "--tmp-dir",
        help="path to store temporary files",
    )
    parser.add_argument(
        "--max-trials",
        help="how many optimizations to try",
        default=MAX_TRIALS,
        type=int,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="print diagnostic info (can be specified more than once)",
        action="count",
        default=1,
    )
    parser.add_argument(
        "file",
        help="code to optimize (can be '-' for stdin)",
        nargs="?",
        default="-",
        metavar="FILE")

    args = parser.parse_args()

    if not args.gem5_path:
        args.gem5_path = os.environ.get("GEM5_PATH")
        error_if(not args.gem5_path, "no GEM5 path specified")

    error_if(
        not (0 <= args.explore_factor <= 1), "exploration factor must be within [0, 1]"
    )

    EXPLORE_FACTOR = args.explore_factor
    FANOUT = args.fanout
    MAX_TRIALS = args.max_trials
    RETRIES = args.retries

    global VERBOSE
    VERBOSE = args.verbose

    if args.quiet:
        VERBOSE = 0

    if args.seed:
        random.seed(int(args.seed))

    # Work in temp directory

    args.gem5_path = Path(args.gem5_path).resolve()
    if args.file != "-":
        args.file = Path(args.file).resolve()
    else:
        args.file = make_temp(sys.stdin.read(), "kernel.c")

    global TMP

    if args.tmp_dir is not None:
        TMP = os.path.join(args.tmp_dir, "optimizer_agent")
        if os.path.exists(TMP):
            shutil.rmtree(TMP)
        os.makedirs(TMP, exist_ok=True)
    else:
        TMP = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(TMP))

    TMP = str(Path(TMP).resolve())

    os.chdir(TMP)

    # Initialize target and do basic checks

    target = AArch64Target(args.gem5_path)
    if VERBOSE:
        print("Target initialized successfully")

    try:
        target.compile(args.file)
    except Target.CompileError as e:
        error(f"failed to compile input kernel:\n{e.err}")

    kernel_name = get_kernel_name("a.out")
    if VERBOSE:
        print(f"Found kernel function {kernel_name}")

    # Initialize LLM and do basic checks

    if args.model == "dummy":
        llm = DummyLLM()
    else:
        short_names = {
            "haiku": "claude-haiku-4.5",
            "sonnet": "claude-sonnet-4.6",
            "opus": "claude-opus-4.7",
        }
        args.model = short_names.get(args.model, args.model)
        llm = AnthropicLLM(args.model)

    llm.ask("Hello, Claude", code_only=False)

    if VERBOSE:
        print(f"LLM '{args.model}' connected successfully")

    # Setup tests

    kernel_code = read_file(args.file)

    test_code = generate_test(kernel_name, kernel_code, target, llm)

    if test_code is None:
        print("Failed to generate testcase for kernel")
        return

    if VERBOSE > 1:
        print(
            f"""\
Generated test:
```
{test_code}
```
"""
        )
    elif VERBOSE:
        print("Generated test")

    test_filename = make_temp(
        f"""\
{test_code}

int main() {{
    test();
    return 0;
}}
""",
        "test.c",
    )

    bench_code, harness_code = generate_bench(kernel_name, kernel_code, target, llm)

    if bench_code is None:
        print("Failed to generate benchmark for kernel")
        return

    if VERBOSE > 1:
        print(
            f"""\
Generated benchmark:
```
{bench_code}
```
"""
        )
    elif VERBOSE:
        print("Generated benchmark")

    bench_filename = make_temp(
        f"""\
{bench_code}

{harness_code}

int main() {{
    bench_N();
    return 0;
}}
""",
        "bench.c",
    )

    # Run optimization loop

    code, improvement = main_loop(
        kernel_name, kernel_code, bench_code, test_filename, bench_filename, target, llm
    )

    if code is None:
        print("Failed to optimize code")
        return

    print(
        f"""\
Optimized code by {100 * improvement:.1f}%:
{code}
"""
    )


if __name__ == "__main__":
    sys.exit(main())
