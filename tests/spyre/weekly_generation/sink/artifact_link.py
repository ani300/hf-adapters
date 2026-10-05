"""Tie a weekly scan's v2 capability verdicts to the image the scan ran on.

One ``artifact_results`` leg per scan run, written after every shard has flushed. The
``spyre`` verdicts depend on that image, so without the leg a tag or artifact page cannot
reach them. Run once per scan, not per shard: the leg is keyed on (artifact, run, test type),
and shards finish in any order.

    python -m tests.spyre.weekly_generation.sink.artifact_link --artifact-id <record>
"""

from __future__ import annotations

import argparse
import os
import sys

from tests.spyre.weekly_generation.sink.capability_write import (
    ARCH,
    COMPONENT,
    TEST_TYPE,
)


def parse_record(raw: str) -> tuple[str, str, str]:
    """derive-gha-artifact-id's ``<artifact_id>|<base_artifact_id>|<installed>`` record."""
    parts = [f.strip() for f in (raw or "").split("|")]
    parts += [""] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def link(client, db: str, record: str, env: dict[str, str]) -> bool:
    """Write the scan's leg; returns whether a row was written."""
    from spyre_clickhouse_ingest import insert_gha_artifact_result, run_id_of

    artifact_id, base_id, installed = parse_record(record)
    gha_run_id = env.get("GITHUB_RUN_ID", "")
    if not (artifact_id and gha_run_id):
        print("  v2: no artifact id or GITHUB_RUN_ID -- no artifact leg written.")
        return False
    run_id = run_id_of("gha", gha_run_id, ARCH, TEST_TYPE)
    verdicts = client.query(
        "SELECT count() FROM {db:Identifier}.capability_runs "
        "WHERE run_id = {run_id:UUID} AND component = {c:String} AND test_type = {t:String}",
        parameters={"db": db, "run_id": run_id, "c": COMPONENT, "t": TEST_TYPE},
    ).result_rows[0][0]
    repo = env.get("GITHUB_REPOSITORY", "")
    server = env.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    wrote = insert_gha_artifact_result(
        client,
        db,
        artifact_id=artifact_id,
        component=COMPONENT,
        arch=ARCH,
        run_id=run_id,
        test_type=TEST_TYPE,
        # A survey, not a gate: most Hub models are expected to fail, so any verdict at all
        # is a completed scan. 'error' marks a scan that recorded nothing.
        state="passed" if verdicts else "error",
        result_kind="capability",
        base_artifact_id=base_id,
        installed=installed,
        repo=repo,
        git_ref=env.get("GITHUB_REF_NAME", ""),
        git_sha=env.get("GITHUB_SHA", ""),
        run_url=f"{server}/{repo}/actions/runs/{gha_run_id}" if repo else "",
        attempt=int(env.get("GITHUB_RUN_ATTEMPT", "0") or 0),
    )
    print(
        f"  v2: artifact leg {artifact_id} [{TEST_TYPE}] over {verdicts} verdict(s) "
        f"under run_id={run_id}: {'written' if wrote else 'not written'}"
    )
    return bool(wrote)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-id", default="")
    args = parser.parse_args()

    from spyre_clickhouse_ingest import get_client, tables_present, target_database
    from spyre_clickhouse_ingest.schema import (
        ARTIFACT_RESULTS,
        ARTIFACTS,
        CAPABILITY_RUNS,
    )

    db = target_database()
    if not db:
        print("  v2: no v2 database configured -- skipping.")
        return 0
    client = get_client()
    if not tables_present(client, db, (ARTIFACTS, ARTIFACT_RESULTS, CAPABILITY_RUNS)):
        print(f"  v2: {db} lacks the artifact tables -- skipping.")
        return 0
    link(client, db, args.artifact_id, dict(os.environ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
