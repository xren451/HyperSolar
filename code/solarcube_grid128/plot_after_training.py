import os, pathlib, subprocess, time, json, datetime
RUN=pathlib.Path(os.environ["HYPERSOLAR_RUN_DIR"])
PLOT=pathlib.Path(__file__).with_name("plot_smf_results_9x24.py")
PYTHON=os.environ.get("PYTHON", "python")
PID=int(os.environ["HYPERSOLAR_TRAIN_PID"])
LOG=RUN/"plot_watcher.log"
def log(event, **kwargs):
    line=json.dumps({"time":datetime.datetime.now().astimezone().isoformat(timespec="seconds"),"event":event,**kwargs})+"\n"
    with LOG.open("a") as f: f.write(line)
while True:
    proc=pathlib.Path(f"/proc/{PID}")
    alive=False
    if proc.exists():
        try: alive=b"train_v9_ramp_hg128.py" in (proc/"cmdline").read_bytes()
        except (OSError,PermissionError): alive=True
    if alive:
        time.sleep(600)
        continue
    metrics=RUN/"metrics.json"
    if metrics.exists() and (RUN/"test_predictions").is_dir():
        log("training_finished_starting_figures")
        with (RUN/"plot_console.log").open("w") as f:
            rc=subprocess.run([PYTHON,"-u",str(PLOT)],stdout=f,stderr=subprocess.STDOUT,check=False).returncode
        log("figures_finished" if rc==0 else "figure_generation_failed", return_code=rc)
    else:
        log("training_process_ended_without_complete_metrics")
    break
