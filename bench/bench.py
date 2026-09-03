#!/usr/bin/env python3
"""torch-side benchmark: our stages (torch.ops.fused_attn.forward_stage) vs PyTorch baselines.

Protocol (docs/BENCHMARK_PROTOCOL.md; identical to bench_main.cu, judge_plan §4.4):
  * >=10 warmup, >=100 timed iterations (or >=500 ms), one cudaEvent pair PER iteration,
    a single synchronize after the loop, median with p10/p90;
  * a 256 MB int32 buffer zeroed before EVERY timed iteration (Triton do_bench convention)
    so no impl benefits from a warm L2; 4 rotating input sets so no impl benefits from
    address reuse either;
  * output checksummed into a sink OUTSIDE the event pair (a discarded output could be
    lazily skipped by a future backend; the sink makes the work observable);
  * a row is written only after the same process passed the correctness gate (fp64 MATH
    reference; ratio rule err_ours <= 2*err_torch_same_dtype + 1e-6 plus per-dtype caps);
  * FLOPs = 4*B*H*N^2*D (causal rows keep the full count and say `flops=full`),
    GB/s from the modelled-minimum bytes in bench/configs.yaml (fused vs materializing impls);
  * the SM clock is sampled DURING the timed loop from a background thread (pynvml, or
    nvidia-smi as the fallback): `clock_mhz` is the median, `clock=<min>-<max>` the spread,
    `noisy_clock` when the spread exceeds 10%; with `--clock-locked <mhz>` the lock is verified
    against the samples and dropped (row says `clock=unlocked`) if the GPU did not honour it;
  * p90/p10 >= NOISE_RATIO -> `noisy` (the same constant bench_main.cu and the sanity test use).

Every row carries commit (`-dirty` if a tracked source file differs from HEAD), gpu, driver,
nvcc, torch, ncu, clock and a `run=<utc>` stamp, so nothing in a README table can be quoted
without its provenance (CONTRACT §3). A refused baseline backend yields a row with
`backend_refused:<message>` and no timing (rule 7), never a silent omission.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import subprocess
import sys
import threading
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fa_common import (  # noqa: E402
    BASELINE_IMPLS,
    CLOCK_SPREAD,
    NOISE_RATIO,
    OUR_IMPLS,
    append_rows,
    attention_flops,
    git_commit,
    latest_rows,
    parse_cfg,
    percentile,
    read_rows,
    stage_csvs,
    utc_stamp,
    write_rows,
)
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

TORCH_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ERR_CAP = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 1e-2}  # absolute caps, docs/NUMERICS.md
L2_FLUSH_BYTES = 256 * 1024 * 1024
N_ROTATE = 4
LOCK_TOLERANCE = 0.02  # a verified lock: median sample within 2% of the requested MHz


# ------------------------------------------------------------------ environment ----------
def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def sample_clock_mhz() -> str:
    """One `clocks.sm` sample via nvidia-smi (0.2-1 s on Colab: use ClockSampler for loops)."""
    out = _run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
    return out.splitlines()[0].strip() if out else ""


class ClockSampler:
    """SM clock samples taken WHILE the GPU executes the timed loop.

    Why a thread: the launch queue is bounded, so by the time the host returns from enqueuing
    the loop the GPU backlog may be only a few hundred launches; a single nvidia-smi call after
    that (0.2-1 s) can land on the idle clock (T4: 300 MHz). pynvml's nvmlDeviceGetClockInfo
    costs microseconds and is sampled every 20 ms from start() until stop() (called after the
    synchronize, so the samples span the whole loop). nvidia-smi is the fallback when pynvml is
    missing; it is slow but still runs concurrently with the loop.
    """

    def __init__(self, interval_s: float = 0.02):
        self.interval = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self.source = "nvidia-smi"
        try:
            import pynvml  # ships as nvidia-ml-py with the CUDA torch wheels

            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            self.source = "pynvml"
        except Exception:  # ImportError or NVML init failure: fall back to the subprocess
            self._nvml = None

    def sample_once(self) -> int | None:
        if self._nvml is not None:
            pynvml, handle = self._nvml
            try:
                return int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            except Exception:
                return None
        s = sample_clock_mhz()
        return int(s) if s.isdigit() else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.sample_once()
            if v:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return list(self.samples)


def clock_notes(samples: list[int], locked_mhz: int | None) -> tuple[str, list[str], bool]:
    """(clock_mhz column, notes, locked_verified) from the samples of one timed loop.

    `locked_mhz` None = no lock requested; 0 = lock asserted by the caller without a value
    (trusted as before); > 0 = the value nvidia-smi -lgc was given, verified here: the median
    sample must be within LOCK_TOLERANCE of it and the spread within CLOCK_SPREAD, otherwise
    the row is labelled clock=unlocked and the caller's assertion is dropped for this row."""
    if not samples:
        return "", (["clock=unlocked"] if not locked_mhz else []), bool(locked_mhz)
    s = sorted(samples)
    med, lo, hi = int(percentile(s, 50)), s[0], s[-1]
    notes = []
    spread = (hi - lo) / med if med else 0.0
    if lo != hi:
        notes.append(f"clock={lo}-{hi}")
    if spread > CLOCK_SPREAD:
        notes.append("noisy_clock")
    if locked_mhz is None:
        return str(med), ["clock=unlocked", *notes], False
    if locked_mhz == 0:
        return str(med), notes, True
    ok = abs(med - locked_mhz) <= LOCK_TOLERANCE * locked_mhz and spread <= CLOCK_SPREAD
    if ok:
        return str(med), [f"clock_lock={locked_mhz}", *notes], True
    return str(med), ["clock=unlocked", f"clock_lock_failed={locked_mhz}", *notes], False


def env_columns() -> dict:
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"])
    parts = [p.strip() for p in smi.splitlines()[0].split(",")] if smi else []
    # `Cuda compilation tools, release 12.8, V12.8.93`: the full version, the same string
    # fa_bench stamps from __CUDACC_VER_*, so bench_to_md's env line never lists two nvcc's.
    nvcc = re.search(r"V(\d+\.\d+\.\d+)", _run(["nvcc", "--version"]))
    ncu = re.search(r"Version ([\d.]+)", _run(["ncu", "--version"]))
    return {
        "commit": git_commit(ROOT),
        "gpu": parts[0] if parts else torch.cuda.get_device_name(0),
        "driver": parts[1] if len(parts) > 1 else "",
        "nvcc": nvcc.group(1) if nvcc else "missing",
        "torch": torch.__version__,
        "ncu": ncu.group(1) if ncu else "missing",
        "clock_mhz": re.sub(r"\D", "", parts[2]) if len(parts) > 2 else "",
    }


def note_env(gpu_dir: Path, line: str) -> None:
    """Appends a provenance line to <gpu>/env.txt once (e.g. the SDPA FLASH refusal text)."""
    gpu_dir.mkdir(parents=True, exist_ok=True)
    p = gpu_dir / "env.txt"
    if p.exists() and line in p.read_text():
        return
    with open(p, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def pin_fp32_ieee() -> str:
    """No TF32 in an fp32 row (rule 7). torch >= 2.9 has the `fp32_precision` knobs and warns
    on the `allow_tf32` booleans; older torch has only the booleans. Returns what was set."""
    mm, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    if hasattr(mm, "fp32_precision"):
        mm.fp32_precision = "ieee"
        if hasattr(cudnn, "conv") and hasattr(cudnn.conv, "fp32_precision"):
            cudnn.conv.fp32_precision = "ieee"
        return "fp32_precision=ieee"
    mm.allow_tf32 = False  # honest fp32 baselines (no TF32 on T4 anyway)
    cudnn.allow_tf32 = False
    return "allow_tf32=False"


def load_configs() -> dict:
    import yaml

    with open(ROOT / "bench" / "configs.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ math -----------------
def causal_mask(N: int, device) -> torch.Tensor:
    # True above the diagonal = masked; top-left aligned like torch is_causal for L == S.
    return torch.ones(N, N, dtype=torch.bool, device=device).triu(1)


def unfused(q, k, v, causal: bool, scale: float, mask=None):
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(
            mask if mask is not None else causal_mask(q.shape[-2], q.device), float("-inf")
        )
    return s.softmax(-1) @ v


def reference_fp64(q, k, v, causal: bool, scale: float):
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), is_causal=causal, scale=scale
        )


def correctness_gate(out, q, k, v, causal, scale, dtype, tol_scale) -> tuple[bool, str]:
    """fp64 oracle on a subsample of (b, h) pairs — attention is independent per head, so
    checking slices of the full-tensor output is exact, and it keeps the fp64 NxN matrix
    (materialized by the MATH backend) small enough for the largest sweep shapes."""
    B, H, _, _ = q.shape
    pairs = sorted({(0, 0), (B - 1, H - 1), (B // 2, H // 2)})
    worst = worst_torch = 0.0
    for b, h in pairs:
        qs, ks, vs = (t[b : b + 1, h : h + 1] for t in (q, k, v))
        ref = reference_fp64(qs, ks, vs, causal, scale)
        mine = out[b : b + 1, h : h + 1]
        if not bool(torch.isfinite(mine).all()):
            return False, f"non-finite output at (b={b},h={h})"
        worst = max(worst, (mine.double() - ref).abs().max().item())
        worst_torch = max(
            worst_torch, (unfused(qs, ks, vs, causal, scale).double() - ref).abs().max().item()
        )
    tol = tol_scale * worst_torch + 1e-6
    cap = ERR_CAP[dtype]
    msg = f"max_err={worst:.3e} torch_same_dtype_err={worst_torch:.3e} tol={tol:.3e} cap={cap:.0e}"
    return (worst <= tol and worst <= cap), msg


# ------------------------------------------------------------------ impls ----------------
def _first_line(e: BaseException, warns) -> str:
    msgs = [str(w.message).splitlines()[0] for w in warns] + [str(e).splitlines()[0]]
    return " | ".join(m for m in msgs if m)[:300]


def _probe(fn) -> str | None:
    """Runs fn once; returns None on success, else the refusal text (exception + warnings)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn(0)
            torch.cuda.synchronize()
        except (RuntimeError, NotImplementedError, TypeError) as e:
            return _first_line(e, w)
    return None


def build_impls(sets, dtype, causal, scale, stages, baselines, gpu_dir):
    """Returns ({impl: fn(set_index) -> out[B,H,N,D]}, {impl: skip reason}, {impl: notes})."""
    impls, skips, notes = {}, {}, {}
    q0 = sets[0][0]
    mask = causal_mask(q0.shape[-2], q0.device) if causal else None

    if "torch_unfused" in baselines:
        impls["torch_unfused"] = lambda i: unfused(*sets[i], causal, scale, mask)
    if "sdpa_math_fp32" in baselines:
        sets32 = sets if dtype == "fp32" else [tuple(t.float() for t in s) for s in sets]

        def math32(i):
            with sdpa_kernel([SDPBackend.MATH]):
                return F.scaled_dot_product_attention(*sets32[i], is_causal=causal, scale=scale)

        impls["sdpa_math_fp32"] = math32
        notes["sdpa_math_fp32"] = ["oracle_not_speed_baseline"]
    for name, attr in (
        ("sdpa_efficient", "EFFICIENT_ATTENTION"),
        ("sdpa_flash", "FLASH_ATTENTION"),
        ("sdpa_cudnn", "CUDNN_ATTENTION"),
    ):
        if name not in baselines:
            continue
        backend = getattr(SDPBackend, attr, None)
        if backend is None:
            skips[name] = f"SDPBackend.{attr} not in torch {torch.__version__}"
            continue

        def sdpa(i, backend=backend):
            with sdpa_kernel([backend]):
                return F.scaled_dot_product_attention(*sets[i], is_causal=causal, scale=scale)

        reason = _probe(sdpa)
        if reason is None:
            impls[name] = sdpa
        else:
            skips[name] = reason
            note_env(gpu_dir, f"{name}: refused ({dtype}): {reason}")
    if "flash_attn" in baselines:
        try:
            from flash_attn import flash_attn_func  # type: ignore
        except ImportError as e:
            skips["flash_attn"] = f"not importable: {str(e).splitlines()[0]}"
        else:
            bnhd = [
                tuple(t.transpose(1, 2).contiguous() for t in s) for s in sets
            ]  # its native layout

            def fa2(i):
                return flash_attn_func(*bnhd[i], causal=causal, softmax_scale=scale).transpose(1, 2)

            reason = _probe(fa2)
            if reason is None:
                impls["flash_attn"] = fa2
                notes["flash_attn"] = ["layout=bnhd"]
            else:
                skips["flash_attn"] = reason
    for stage in stages:
        name = OUR_IMPLS[stage]
        if not hasattr(torch.ops, "fused_attn") or not hasattr(
            torch.ops.fused_attn, "forward_stage"
        ):
            skips[name] = (
                "torch.ops.fused_attn.forward_stage missing (extension not built/imported)"
            )
            continue

        def ours(i, stage=stage):
            return torch.ops.fused_attn.forward_stage(*sets[i], causal, scale, stage)[0]

        reason = _probe(ours)
        if reason is None:
            impls[name] = ours
        else:
            skips[name] = f"unsupported on this build/device: {reason}"
    return impls, skips, notes


# ------------------------------------------------------------------ timing ---------------
def time_impl(
    fn, warmup: int, iters: int, min_ms: float, max_iters: int, flush, sink, sampler: ClockSampler
) -> dict:
    for w in range(warmup):
        fn(w % N_ROTATE)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(3):
        fn(i % N_ROTATE)
    e.record()
    torch.cuda.synchronize()
    est_ms = max(s.elapsed_time(e) / 3.0, 1e-3)
    iters = int(min(max_iters, max(iters, math.ceil(min_ms / est_ms))))  # >=100 iters or >=500 ms
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    out = None
    sampler.start()  # samples the SM clock every 20 ms until the loop has drained
    for i in range(iters):
        flush.zero_()  # cold L2 for every timed iteration
        starts[i].record()
        out = fn(i % N_ROTATE)
        ends[i].record()
        sink.add_(out.reshape(-1)[:1024].double().sum())  # after the end event: not timed
    torch.cuda.synchronize()
    clocks = sampler.stop()
    times = sorted(a.elapsed_time(b) for a, b in zip(starts, ends, strict=True))
    return {
        "median": percentile(times, 50),
        "p10": percentile(times, 10),
        "p90": percentile(times, 90),
        "iters": iters,
        "clocks": clocks,
        "out": out,
    }


def intermediate_bytes(icfg: dict, dtype: str) -> int:
    """Element size of a materializing impl's NxN S/P: fp32 for our workspace and the MATH
    oracle, the input dtype for torch_unfused (configs.yaml `intermediate`)."""
    if icfg.get("intermediate", "fp32") == "fp32":
        return 4
    return 4 if dtype == "fp32" else 2


def bytes_moved(icfg: dict, B, H, N, D, dtype: str) -> float:
    """Modelled MINIMUM bytes (BENCHMARK_PROTOCOL rule 6): Q+K+V+O (+LSE), plus one write
    and one read each of S and P for materializing impls. torch's unfused path makes extra
    NxN passes (scale, mask); the measured DRAM bytes are the NCU table's `dram__bytes.sum`."""
    el = 4 if (dtype == "fp32" or icfg.get("io") == "fp32") else 2
    total = 4.0 * B * H * N * D * el + (4.0 * B * H * N if icfg.get("writes_lse") else 0.0)
    if icfg.get("materializing"):
        total += 4.0 * B * H * N * N * intermediate_bytes(icfg, dtype)  # S w, S r, P w, P r
    return total


def oom_by_design(name: str, oom: dict, B, H, N, total_mem: int, inter_bytes: int = 4) -> bool:
    """S and P together (2*B*H*N*N*inter_bytes) above `fraction` of cudaDeviceProp
    totalGlobalMem: the same rule as bench_main.cu (`workspace_bytes > total_bytes / 4`)."""
    if name not in oom["applies_to"]:
        return False
    return 2.0 * B * H * N * N * inter_bytes > oom["fraction"] * total_mem


# ------------------------------------------------------------------ main -----------------
def resolve_shapes(args, cfgs) -> list[dict]:
    if args.sweep:
        sw = cfgs["sweep"]
        return [
            dict(B=b, H=h, N=n, D=d)
            for n, d, (b, h) in itertools.product(sw["N"], sw["D"], sw["BH"])
        ]
    shapes = []
    for tok in args.cfg.split(","):
        tok = tok.strip()
        if tok == "canonical":
            shapes += [{k: v[k] for k in "BHND"} for v in cfgs["canonical"].values()]
        elif tok in cfgs["canonical"]:
            shapes.append({k: cfgs["canonical"][tok][k] for k in "BHND"})
        else:
            shapes.append(parse_cfg(tok))
    return shapes


def resolve_causals(args, cfgs) -> list[int]:
    """`--causal` wins; a `--sweep` without it takes `sweep.causal` from configs.yaml (CONTRACT
    §5: the 48-config sweep includes causal 0 and 1); otherwise causal 0."""
    if args.causal is not None:
        return [0, 1] if args.causal == "both" else [int(args.causal)]
    if args.sweep:
        return [int(c) for c in cfgs["sweep"].get("causal", [0])]
    return [0]


def rebuild_summary(gpu_dir: Path) -> None:
    rows = []
    for f in stage_csvs(gpu_dir.parent).get(gpu_dir.name, []):
        rows += read_rows(f)[1]
    if rows:
        write_rows(gpu_dir / "summary.csv", latest_rows(rows))


def clean_note(text: str) -> str:
    """Free text into the `;`-separated notes column: no separators, no newlines."""
    return re.sub(r"[;\r\n]+", " ", text).strip()[:200]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", default="1", help="1..5, 'all', or 0 = baselines only")
    ap.add_argument("--gpu", required=True, help="results dir name, e.g. t4 (CONTRACT §3)")
    ap.add_argument("--out", help="CSV path; default bench/results/<gpu>/stage<N>.csv")
    ap.add_argument(
        "--cfg", default="canonical", help="C64,C128,canonical or B2_H8_N2048_D64 (comma list)"
    )
    ap.add_argument(
        "--sweep", action="store_true", help="run the CONTRACT §5 sweep instead of --cfg"
    )
    ap.add_argument(
        "--dtype", help="fp32|fp16|bf16 (comma list); default per stage from configs.yaml"
    )
    ap.add_argument(
        "--causal",
        default=None,
        choices=["0", "1", "both"],
        help="default: 0, or configs.yaml sweep.causal (both) with --sweep",
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--min-ms", type=float, default=500.0, help="timed loop lasts at least this long"
    )
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument(
        "--baselines", default="all", help="comma list of baseline impls, 'all' or 'none'"
    )
    ap.add_argument(
        "--tol-scale", type=float, default=2.0, help="ratio-rule multiplier (2 = FA test suite)"
    )
    ap.add_argument(
        "--clock-locked",
        nargs="?",
        const=0,
        default=None,
        type=int,
        metavar="MHZ",
        help="clocks locked via nvidia-smi -lgc; with a value the lock is verified per row",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("bench.py: CUDA not available; nothing to measure", file=sys.stderr)
        return 2
    precision = pin_fp32_ieee()
    cfgs = load_configs()
    gpu_dir = ROOT / "bench" / "results" / args.gpu
    gpu_dir.mkdir(parents=True, exist_ok=True)
    note_env(gpu_dir, f"bench.py fp32 policy: {precision} (torch {torch.__version__})")
    stages = (
        [] if args.stage == "0" else (list(OUR_IMPLS) if args.stage == "all" else [int(args.stage)])
    )
    ext_err = None
    try:
        import fused_attn  # noqa: F401  (registers torch.ops.fused_attn)
    except Exception as e:  # ImportError or a CUDA load failure — report, still run baselines
        ext_err = str(e).splitlines()[0] if str(e) else repr(e)
        print(f"bench.py: fused_attn extension not importable: {ext_err}")
    baselines = (
        []
        if args.baselines == "none"
        else (BASELINE_IMPLS if args.baselines == "all" else args.baselines.split(","))
    )
    if not baselines:
        print("[bench] WARNING: --baselines none: rows will have no SDPA denominator")
    dtypes = (
        args.dtype.split(",")
        if args.dtype
        else sorted({cfgs["stage_dtype"].get(s, "fp32") for s in stages} or {"fp32"})
    )
    causals = resolve_causals(args, cfgs)
    env = env_columns()
    run_id = utc_stamp()
    sampler = ClockSampler()
    note_env(gpu_dir, f"bench.py clock sampling: {sampler.source} (background thread, 20 ms)")
    device = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device)
    sink = torch.zeros((), dtype=torch.float64, device=device)
    produced_ours = 0
    lock_failures = 0
    for shape, causal, dtype in itertools.product(resolve_shapes(args, cfgs), causals, dtypes):
        B, H, N, D = (shape[k] for k in "BHND")
        scale = 1.0 / math.sqrt(D)
        g = torch.Generator(device=device).manual_seed(args.seed)
        sets = [
            tuple(
                torch.randn(B, H, N, D, generator=g, device=device).to(TORCH_DTYPE[dtype])
                for _ in range(3)
            )
            for _ in range(N_ROTATE)
        ]
        impls, skips, inotes = build_impls(
            sets, dtype, bool(causal), scale, stages, baselines, gpu_dir
        )
        rows_by_file: dict[Path, list[dict]] = {}

        def out_path(stage: int) -> Path:
            if args.out:
                return Path(args.out)
            return gpu_dir / f"stage{stage if stage else (stages[0] if stages else 0)}.csv"

        common = []  # the FLOP convention is per impl (materialising -> full, skipping -> half)
        if args.tol_scale != 2.0:
            common.append(f"tol_scale={args.tol_scale:g}")
        common.append(f"run={run_id}")
        for name, why in skips.items():
            print(f"[bench] skip impl={name} cfg=B{B}_H{H}_N{N}_D{D} dtype={dtype}: {why}")
            if name in BASELINE_IMPLS:  # rule 7: a refused backend is a row, never an omission
                stage = cfgs["impls"][name]["stage"]
                rows_by_file.setdefault(out_path(stage), []).append(
                    dict(
                        env,
                        impl=name,
                        stage=stage,
                        dtype=dtype,
                        B=B,
                        H=H,
                        N=N,
                        D=D,
                        causal=causal,
                        clock_mhz="",
                        notes=";".join([*common, f"backend_refused:{clean_note(why)}"]),
                    )
                )
        for name in [*OUR_IMPLS.values(), *BASELINE_IMPLS]:
            if name not in impls:
                continue
            icfg = cfgs["impls"][name]
            stage = icfg["stage"]
            out_csv = out_path(stage)
            notes = list(inotes.get(name, [])) + common
            base = dict(env, impl=name, stage=stage, dtype=dtype, B=B, H=H, N=N, D=D, causal=causal)
            inter = intermediate_bytes(icfg, dtype)
            if oom_by_design(name, cfgs["oom_by_design"], B, H, N, total_mem, inter):
                print(
                    f"[bench] oom_by_design impl={name} cfg=B{B}_H{H}_N{N}_D{D} "
                    f"({'fp32' if inter == 4 else dtype} NxN S+P > "
                    f"{cfgs['oom_by_design']['fraction']:.0%} of device memory)"
                )
                rows_by_file.setdefault(out_csv, []).append(
                    dict(base, clock_mhz="", notes=";".join(notes + ["oom_by_design"]))
                )
                continue
            try:
                out = impls[name](0)
                torch.cuda.synchronize()
                ok, msg = correctness_gate(
                    out, *sets[0], bool(causal), scale, dtype, args.tol_scale
                )
                if not ok:
                    print(
                        f"[bench] CORRECTNESS FAILED impl={name} cfg=B{B}_H{H}_N{N}_D{D} "
                        f"dtype={dtype} causal={causal}: {msg}"
                    )
                    failed = ";".join(notes + ["correctness_failed", msg])
                    rows_by_file.setdefault(out_csv, []).append(
                        dict(base, clock_mhz="", notes=failed)
                    )
                    continue
                del out
                t = time_impl(
                    impls[name],
                    args.warmup,
                    args.iters,
                    args.min_ms,
                    args.max_iters,
                    flush,
                    sink,
                    sampler,
                )
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                first = str(e).splitlines()[0][:120] if str(e) else repr(e)
                print(f"[bench] OOM at runtime impl={name} cfg=B{B}_H{H}_N{N}_D{D}: {first}")
                rows_by_file.setdefault(out_csv, []).append(
                    dict(base, clock_mhz="", notes=";".join(notes + ["oom_runtime"]))
                )
                continue
            ms = t["median"]
            # Protocol rule 5: a causal row of a kernel that skips fully masked tiles is
            # accounted at half the FLOPs (FlashAttention convention); materialising impls
            # (S1/S2, torch_unfused, the MATH oracle) still do the full N^2 work.
            causal_half = bool(causal) and not icfg.get("materializing", False)
            flops = attention_flops(B, H, N, D, causal_half=causal_half)
            if causal:
                notes = notes + ["flops=causal_half" if causal_half else "flops=full"]
            nbytes = bytes_moved(icfg, B, H, N, D, dtype)
            clock_col, cnotes, locked_ok = clock_notes(t["clocks"], args.clock_locked)
            if args.clock_locked and not locked_ok:
                lock_failures += 1
            if t["p10"] > 0 and t["p90"] / t["p10"] >= NOISE_RATIO:
                cnotes.append("noisy")
            row = dict(
                base,
                clock_mhz=clock_col or env["clock_mhz"],
                ms_median=f"{ms:.4f}",
                ms_p10=f"{t['p10']:.4f}",
                ms_p90=f"{t['p90']:.4f}",
                tflops=f"{flops / (ms * 1e-3) / 1e12:.3f}",
                gbps=f"{nbytes / (ms * 1e-3) / 1e9:.1f}",
                notes=";".join(notes + cnotes + [f"iters={t['iters']}", msg]),
            )
            rows_by_file.setdefault(out_csv, []).append(row)
            produced_ours += name in OUR_IMPLS.values()
            # One greppable line per row: gpu_run.sh's SUMMARY block is built from these.
            print(
                f"BENCH_ROW impl={name} stage={stage} cfg=B{B}_H{H}_N{N}_D{D} dtype={dtype} "
                f"causal={causal} ms_median={ms:.4f} p10={t['p10']:.4f} p90={t['p90']:.4f} "
                f"tflops={row['tflops']} gbps={row['gbps']} clock={row['clock_mhz']}"
            )
        for path, rows in rows_by_file.items():
            append_rows(path, rows)
        del sets, impls
        torch.cuda.empty_cache()
    print(f"[bench] checksum sink = {sink.item():.6e}")
    if lock_failures:
        print(
            f"[bench] WARNING: --clock-locked {args.clock_locked} was not honoured on "
            f"{lock_failures} row(s); those rows say clock=unlocked"
        )
    rebuild_summary(gpu_dir)
    if stages and ext_err is not None:
        print(
            "bench.py: stage rows requested but the extension is not importable (exit 4)",
            file=sys.stderr,
        )
        return 4
    if stages and produced_ours == 0:
        print("[bench] WARNING: no row for our kernels was produced (unsupported or failed gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
