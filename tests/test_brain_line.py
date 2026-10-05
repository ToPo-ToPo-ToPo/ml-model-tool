"""A brain's line, end to end without a desktop: parse -> check -> job as a CLI process -> slim answer."""
from ml_model_tool_cli.declare import COMMANDS


def test_data_line_runs_as_a_job_and_returns_artifacts(ws):
    res = COMMANDS.type_line("data --spec simp.json", ws)
    assert res["ok"] and res["state"] == "done", res
    assert res["job_dir"].startswith("jobs/") and "_ml-model-tool_data" in res["job_dir"]
    paths = {a["path"] for a in res["artifacts"]}
    assert f"{res['job_dir']}/dataset.npz" in paths and f"{res['job_dir']}/summary.md" in paths
    assert all(not p.startswith("/") for p in paths)          # work-folder relative

    listed = COMMANDS.type_line("job list", ws)
    assert listed["ok"] and listed["jobs"][0]["state"] == "done"

    st = COMMANDS.type_line(f"job status --job-dir {res['job_dir']}", ws)
    assert st["state"] == "done"
