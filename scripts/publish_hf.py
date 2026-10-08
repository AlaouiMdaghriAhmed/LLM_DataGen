"""Publish a synthitd dataset instance to the Hugging Face Hub.

Generates a dataset instance deterministically (or reuses an existing output dir),
drops in the Hugging Face dataset card (hf/README.md), and uploads everything to a
Hub dataset repo. Run it from a machine/environment that has network access to
huggingface.co and an auth token.

Auth (any one):
  * export HF_TOKEN=hf_xxx         (recommended)
  * huggingface-cli login
Install:
  pip install -e ".[hf]"           # adds huggingface_hub

Examples:
  # generate a fresh demo instance and push it
  python scripts/publish_hf.py --repo-id <user>/synthitd --domain tech \
      --employees 60 --days 60 --seed 7

  # push an already-generated directory as-is
  python scripts/publish_hf.py --repo-id <user>/synthitd --from-dir out/demo_tech

  # make it public (default is private)
  python scripts/publish_hf.py --repo-id <user>/synthitd --from-dir out/demo_tech --public
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = os.path.join(REPO_ROOT, "hf", "README.md")


def _generate(args) -> str:
    """Materialise a dataset instance into args.from_dir and return the path."""
    sys.path.insert(0, REPO_ROOT)
    from synthitd.config import SimConfig, load_config
    from synthitd.simulate import simulate
    from synthitd.render import render_events
    from synthitd.writers import write_dataset
    from synthitd.export_cert import export_cert

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SimConfig(
            name=f"synthitd_{args.domain}", domain=args.domain,
            n_employees=args.employees, horizon_days=args.days, seed=args.seed,
            insider_prevalence=args.prevalence, renderer=args.renderer,
        )
    print(f"[hf] generating {cfg.name}: {cfg.domain}, {cfg.n_employees} emp, "
          f"{cfg.horizon_days} d, seed {cfg.seed} ...", flush=True)
    res = simulate(cfg)
    rr = render_events(res.events, res.org, cfg) if not args.no_render else None
    out = args.from_dir or os.path.join(REPO_ROOT, "out", cfg.name)
    write_dataset(res, out, rr)
    export_cert(res, os.path.join(out, "cert"))
    print(f"[hf] wrote dataset to {out} ({len(res.events):,} events, "
          f"{len(res.insiders())} activated insiders)", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True, help="Hub dataset id, e.g. <user>/synthitd")
    p.add_argument("--from-dir", help="use/emit the dataset here (skip regen if it exists)")
    p.add_argument("--config", help="SimConfig YAML/JSON to generate from")
    p.add_argument("--domain", default="tech", choices=["tech", "finance", "healthcare"])
    p.add_argument("--employees", type=int, default=60)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prevalence", type=float, default=0.10)
    p.add_argument("--renderer", default="template", choices=["template", "anthropic"])
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--public", action="store_true", help="create a public repo (default private)")
    p.add_argument("--clean", action="store_true",
                   help="make the Hub repo exactly mirror the folder (delete stale files)")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token (default: $HF_TOKEN or cached login)")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except Exception:
        print("[hf] huggingface_hub not installed. Run: pip install -e '.[hf]'", file=sys.stderr)
        return 2

    # If --from-dir points at a prepared dataset, upload it AS-IS (never regenerate,
    # never touch its card). A prepared folder is either a flat instance (has
    # events.jsonl) or a multi-domain release (has README.md and/or <domain>/events.jsonl).
    data_dir = args.from_dir
    prepared = _looks_prepared(data_dir)
    if prepared:
        print(f"[hf] using prepared dataset at {data_dir} as-is (no generation)", flush=True)
    else:
        if data_dir and os.path.isdir(data_dir) and os.listdir(data_dir):
            print(f"[hf] WARNING: {data_dir} exists but has no dataset files; "
                  f"generating a single instance into it", file=sys.stderr)
        data_dir = _generate(args)
        # only a freshly-generated flat instance gets the single-instance card
        if os.path.exists(CARD):
            shutil.copyfile(CARD, os.path.join(data_dir, "README.md"))
        else:
            print(f"[hf] warning: dataset card {CARD} not found; uploading without it", file=sys.stderr)

    api = HfApi(token=args.token)
    who = api.whoami()  # fails fast with a clear error if the token is bad/missing
    print(f"[hf] authenticated as {who.get('name', '?')}", flush=True)

    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    nfiles = sum(len(fs) for _, _, fs in os.walk(data_dir))
    print(f"[hf] uploading {data_dir} ({nfiles} files) -> {args.repo_id} "
          f"(private={not args.public}, clean={args.clean}) ...", flush=True)
    kwargs = dict(repo_id=args.repo_id, repo_type="dataset", folder_path=data_dir,
                  commit_message="Add synthitd dataset")
    if args.clean:
        kwargs["delete_patterns"] = ["*"]  # remove hub files not present in the folder
    api.upload_folder(**kwargs)
    print(f"[hf] done: https://huggingface.co/datasets/{args.repo_id}", flush=True)
    return 0


def _looks_prepared(d: str | None) -> bool:
    """True if d already holds a dataset (flat instance or multi-domain release)."""
    if not (d and os.path.isdir(d)):
        return False
    if os.path.exists(os.path.join(d, "events.jsonl")):      # flat instance
        return True
    if os.path.exists(os.path.join(d, "README.md")):         # release card present
        return True
    for name in os.listdir(d):                                # any domain subfolder
        if os.path.exists(os.path.join(d, name, "events.jsonl")):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
