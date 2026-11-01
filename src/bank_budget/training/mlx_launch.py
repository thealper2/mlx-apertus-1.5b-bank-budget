"""Run ``mlx_lm lora`` with MLX's memory use bounded for small Macs.

MLX schedules work up to its memory limit, which defaults to 1.5 x the GPU's recommended
working set (8.2 GB on an 8 GB M1, whose GPU working set is 5.7 GB), and it keeps freed
buffers in a cache for reuse. On an 8 GB Mac either can push the total past what Metal
will give the GPU, and the run aborts with "Insufficient Memory". A lower memory limit
makes MLX wait for scheduled work to finish (and release its buffers) before allocating
more; a cache cap returns freed buffers. Both change only how memory is used, never a
computed value.

Usage: ``python -m bank_budget.training.mlx_launch [--memory-limit-gb X]
[--cache-limit-gb Y] -c config.yaml``
"""

from __future__ import annotations

import sys

_OPTIONS = ("--memory-limit-gb", "--cache-limit-gb")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    limits: dict[str, float] = {}
    while args[:1] and args[0] in _OPTIONS:
        limits[args[0]] = float(args[1])
        args = args[2:]
    if limits:
        import mlx.core as mx

        if "--memory-limit-gb" in limits:
            mx.set_memory_limit(int(limits["--memory-limit-gb"] * 1e9))
        if "--cache-limit-gb" in limits:
            mx.set_cache_limit(int(limits["--cache-limit-gb"] * 1e9))
    from mlx_lm import lora

    sys.argv = ["mlx_lm.lora", *args]
    lora.main()


if __name__ == "__main__":
    main()
