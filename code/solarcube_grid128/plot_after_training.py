import os, pathlib, subprocess, time, json, datetime
RUN=pathlib.Path("/mnt/wangyishan_share/wangdongdong/smf_registry_variants_20261105/results/SolarCube/V9_RampHG_Grid128_seed42_GPU4_full19_native120eval")
PLOT=pathlib.Path("/mnt/wangyishan_share/wangdongdong/smf_registry_variants_20261105/code/v9_ramp_hg128_20261110/plot_smf_results_9x24.py")
PYTHON="/home/wangdongdong/miniconda3/envs/geo/bin/python"
PID=2495722
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
