#!/usr/bin/env python3
"""Check native normalization/delivery against Chrome's original frame rendering.

Requires an already-open, task-owned playwright-cli session. The caller owns
opening/closing that session; this script never attaches to an interactive tab.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True, help="Lucid executable")
    parser.add_argument("--session", required=True, help="Existing playwright-cli session")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    template = Path(__file__).with_name("color_probe_browser.js")
    with tempfile.TemporaryDirectory(prefix="lucid-color-") as temporary:
        directory = Path(temporary)
        probe = directory / "probe.json"
        subprocess.run([str(args.app.resolve()), "--color-probe", str(probe)], check=True, timeout=30)
        script = directory / "probe.js"
        script.write_text(template.read_text().replace("__PROBE_DATA__", probe.read_text()))
        result = subprocess.run(
            ["playwright-cli", f"-s={args.session}", "--raw", "run-code", "--filename", str(script)],
            check=True, capture_output=True, text=True, timeout=45,
            env={**os.environ, "PLAYWRIGHT_SKIP_BROWSER_GC": "1"})
        reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"purpose"')]
        if len(reports) != 1:
            raise RuntimeError("Browser returned no unique color report")
        report = reports[0]
    report["native_sha256"] = hashlib.sha256(args.app.read_bytes()).hexdigest()
    report["browser_probe_sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()
    report["limitations"] = ["12 flat patches, 8-bit SDR, Chrome canvas only",
                             "Enhancement bypassed; does not certify HDR, Safari, ICC displays or chroma edges"]
    report["max_allowed_error"] = 4
    report["passed"] = ({case["id"] for case in report["cases"]} == {"srgb", "p3", "rec709"}
                        and all(case["maxError"] <= 4 for case in report["cases"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": {
        case["id"]: case["maxError"] for case in report["cases"]}}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
