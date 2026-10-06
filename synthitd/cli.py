"""Command-line interface for synthitd.

Examples
--------
    python -m synthitd.cli generate --config configs/tech_small.yaml --out out/tech
    python -m synthitd.cli benchmark --data out/tech
    python -m synthitd.cli export-cert --data out/tech --out out/tech/cert
    python -m synthitd.cli info
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import SimConfig, load_config
from .simulate import simulate, SimResult


def _load_result_from_config(cfg: SimConfig) -> SimResult:
    return simulate(cfg)


def cmd_generate(args: argparse.Namespace) -> int:
    from .render import render_events
    from .writers import write_dataset

    cfg = load_config(args.config) if args.config else SimConfig()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.employees is not None:
        overrides["n_employees"] = args.employees
    if args.days is not None:
        overrides["horizon_days"] = args.days
    if args.domain:
        overrides["domain"] = args.domain
    if args.renderer:
        overrides["renderer"] = args.renderer
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    print(f"[generate] simulating '{cfg.name}' "
          f"({cfg.domain}, {cfg.n_employees} emp, {cfg.horizon_days}d, seed={cfg.seed}) ...",
          file=sys.stderr)
    result = simulate(cfg)
    rr = None
    if not args.no_render:
        print(f"[generate] rendering with backend '{cfg.renderer}' ...", file=sys.stderr)
        rr = render_events(result.events, result.org, cfg)
    manifest = write_dataset(result, args.out, rr)
    print(json.dumps(manifest["summary"], indent=2))
    print(f"[generate] wrote dataset to {args.out}", file=sys.stderr)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark.run import run_benchmark

    # regenerate from the saved manifest config (keeps it a pure function of config)
    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    cfg = load_config(manifest["config"])
    result = simulate(cfg)
    report = run_benchmark(result, budget=args.budget, observed=not args.full_stream)
    out_path = os.path.join(args.data, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[benchmark] wrote {out_path}", file=sys.stderr)
    return 0


def cmd_export_cert(args: argparse.Namespace) -> int:
    from .export_cert import export_cert

    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    cfg = load_config(manifest["config"])
    result = simulate(cfg)
    info = export_cert(result, args.out, observed=not args.full_stream)
    print(json.dumps(info, indent=2))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .org import DOMAINS
    from .threats import PLAYBOOKS, ThreatEngine

    print(json.dumps({
        "version": __import__("synthitd").__version__,
        "domains": list(DOMAINS),
        "playbooks": {k: v.description for k, v in PLAYBOOKS.items()},
        "technique_ids": ThreatEngine.all_technique_ids(),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synthitd", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="simulate and write a dataset")
    g.add_argument("--config", help="YAML/JSON config path")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--seed", type=int)
    g.add_argument("--employees", type=int)
    g.add_argument("--days", type=int)
    g.add_argument("--domain", choices=["tech", "finance", "healthcare"])
    g.add_argument("--renderer", choices=["template", "anthropic"])
    g.add_argument("--no-render", action="store_true", help="skip free-text rendering")
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("benchmark", help="run the benchmark on a generated dataset")
    b.add_argument("--data", required=True, help="dataset directory (with manifest.json)")
    b.add_argument("--budget", type=float, default=0.01, help="review budget fraction")
    b.add_argument("--full-stream", action="store_true", help="use full (not observed) stream")
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("export-cert", help="export a CERT r6.2-compatible view")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--full-stream", action="store_true")
    c.set_defaults(func=cmd_export_cert)

    i = sub.add_parser("info", help="print domains, playbooks, technique ids")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
