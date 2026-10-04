"""Run the HATMD manuscript analyses in dependency order."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--mode",
        choices=["all", "core"],
        default="all",
        help="core omits classification, multiple imputation and Figure S1",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare generated aggregate results with manuscript references",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_file():
        parser.error(f"Input file does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent
    tasks = [
        "table1",
        "table2",
        "table_s1",
        "figure1",
        "figure2",
        "figure3",
        "table_s3",
    ]
    if args.mode == "all":
        tasks += ["table3", "table_s2", "figure_s1"]
    tasks += ["export_tables"]
    env = os.environ.copy()
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[key] = "1"
    manifest = {
        "source": str(source),
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "tasks": [],
    }
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task}", flush=True)
        command = [
            sys.executable,
            str(base / "scripts" / f"{task}.py"),
            "--source",
            str(source),
            "--output",
            str(output),
        ]
        start = time.time()
        with (logs / f"{task}.log").open("w") as log:
            process = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        manifest["tasks"].append(
            {
                "task": task,
                "exit_code": process.returncode,
                "elapsed_seconds": round(time.time() - start, 2),
            }
        )
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        if process.returncode:
            print((logs / f"{task}.log").read_text(), file=sys.stderr)
            raise SystemExit(f"{task} failed; see {logs/task}.log")
    if args.verify:
        subprocess.run(
            [
                sys.executable,
                str(base / "verify_outputs.py"),
                "--output",
                str(output),
                "--mode",
                args.mode,
            ],
            check=True,
            env=env,
        )
    print(
        f'Completed. Aggregate tables: {output / "tables"}; figures: {output / "figures"}',
        flush=True,
    )


if __name__ == "__main__":
    main()
