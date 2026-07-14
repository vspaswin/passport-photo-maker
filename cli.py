#!/usr/bin/env python3
"""CLI for Passport Photo Maker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_check(args: argparse.Namespace) -> int:
    from app.engine.process import load_image
    from app.engine.specs import get_spec
    from app.engine.validate import assess_photo

    path = Path(args.image)
    data = path.read_bytes()
    im = load_image(data)
    report = assess_photo(im, get_spec(args.doc_type))
    print(json.dumps(report, indent=2))
    return 0 if report.get("can_convert") or report.get("can_check_only_pass") else 1


def cmd_convert(args: argparse.Namespace) -> int:
    from app.engine.process import process_photo
    from app.engine.validate import PhotoValidationError

    path = Path(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = process_photo(
            path.read_bytes(), doc_type=args.doc_type, remove_bg=True, strict=True
        )
    except PhotoValidationError as exc:
        print(json.dumps({"ok": False, "message": exc.message, "validation": exc.report.to_dict()}, indent=2), file=sys.stderr)
        return 2
    for name, blob in result.files.items():
        (out / name).write_bytes(blob)
        print(f"wrote {out / name}")
    (out / "preview.jpg").write_bytes(result.preview_jpeg)
    print(json.dumps({"ok": True, "files": list(result.files.keys()), "metrics": result.metrics}, indent=2))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from app.engine.process import process_photo
    from app.engine.validate import PhotoValidationError

    src = Path(args.input_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    images = sorted(
        list(src.glob("*.jpg"))
        + list(src.glob("*.jpeg"))
        + list(src.glob("*.png"))
        + list(src.glob("*.JPG"))
        + list(src.glob("*.PNG"))
    )
    summary = []
    for path in images:
        entry = {"file": path.name, "ok": False}
        try:
            result = process_photo(
                path.read_bytes(), doc_type=args.doc_type, remove_bg=True, strict=True
            )
            dest = out_root / path.stem
            dest.mkdir(exist_ok=True)
            for name, blob in result.files.items():
                (dest / name).write_bytes(blob)
            entry.update({"ok": True, "dir": str(dest), "files": list(result.files.keys())})
            print(f"PASS {path.name}")
        except PhotoValidationError as exc:
            entry.update({"message": exc.message, "codes": [i.code for i in exc.report.issues]})
            print(f"FAIL {path.name}: {entry['codes']}")
        except Exception as exc:  # noqa: BLE001
            entry["message"] = str(exc)
            print(f"ERROR {path.name}: {exc}")
        summary.append(entry)
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    passed = sum(1 for s in summary if s.get("ok"))
    print(f"Done: {passed}/{len(summary)} passed → {out_root}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="passport-photo",
        description="Validate and convert photos to Indian passport format",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="As-is + convertible assessment (JSON)")
    p_check.add_argument("image", help="Path to image")
    p_check.add_argument("--doc-type", default="indian-passport")
    p_check.set_defaults(func=cmd_check)

    p_conv = sub.add_parser("convert", help="Convert one image if QC passes")
    p_conv.add_argument("image", help="Path to image")
    p_conv.add_argument("-o", "--out", default="./passport_out", help="Output directory")
    p_conv.add_argument("--doc-type", default="indian-passport")
    p_conv.set_defaults(func=cmd_convert)

    p_batch = sub.add_parser("batch", help="Convert a folder of images")
    p_batch.add_argument("input_dir", help="Directory of images")
    p_batch.add_argument("-o", "--out", default="./passport_batch_out")
    p_batch.add_argument("--doc-type", default="indian-passport")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
