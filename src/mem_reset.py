"""작업 사이 메모리 정리 (2026-10-01): 이전 작업이 읽은 파일의 캐시를 비우고(관리자 권한 불필요, posix_fadvise),
이전 작업 프로세스·GPU 사용이 끝났는지, 쓸 수 있는 메모리가 기준 이상으로 돌아왔는지 확인한 뒤 기록한다.
사용: python mem_reset.py <태그> [--min-avail-gb 80] [--wait-min 20] 경로...
"""
import argparse, os, subprocess, time, datetime
from pathlib import Path
LOG = Path(__file__).resolve().parents[1] / "runs" / "mem_reset.log"
MINE = ("prepare_material.py", "measure_shapes.py", "train_diffusion.py", "make_synth.py", "make_subsets.py", "prepare_real500.py",
        "v1_train.py", "v1_eval.py", "v2_train.py", "v2_eval.py", "v3_train.py", "v3_eval.py", "v4_train.py", "v4_eval.py", "v5_train.py", "v5_eval.py",
        "analyze.py", "compare.py", "test2_compare.py", "make_figures.py", "export_submission.py", "generate.py", "predict.py", "sd.sample")

def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 1048576
def cached_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("Cached:"): return int(l.split()[1]) / 1048576
def my_procs():
    """실행 파일이 python 인 프로세스 중 우리 작업 스크립트를 돌리는 것만 (셸 명령 문자열에 이름이 들어간 껍데기는 제외)"""
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines(); res = []
    for l in out:
        parts = l.split(None, 2)
        if len(parts) < 3 or not os.path.basename(parts[1]).startswith("python"): continue
        toks = parts[2].split(); script = next((t for t in toks if t.endswith(".py")), "")
        if not script and "-m" in toks and toks.index("-m") + 1 < len(toks): script = toks[toks.index("-m") + 1]   # python -m sd.train
        if os.path.basename(script) in MINE: res.append(f"{parts[0]} {os.path.basename(script)}")
    return sorted(set(res))
def gpu_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
    return [l for l in r.stdout.strip().splitlines() if l.strip()]
def evict(paths):
    n = 0
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else (p.rglob("*") if p.exists() else [])
        for f in files:
            if not f.is_file(): continue
            try:
                fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError: pass
    return n
def log(msg):
    line = f"[{datetime.datetime.now():%F %T}] {msg}"; print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)   # 새로 받은 제출본에는 runs/ 가 없음
    with open(LOG, "a") as fh: fh.write(line + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("paths", nargs="*")
    ap.add_argument("--min-avail-gb", type=float, default=80); ap.add_argument("--wait-min", type=float, default=20)
    a = ap.parse_intermixed_args()
    log(f"{a.tag}: 시작  가용 {avail_gb():.1f}GB  파일캐시 {cached_gb():.1f}GB  GPU {gpu_apps() or '없음'}")
    t0 = time.time()
    while my_procs() and time.time() - t0 < a.wait_min * 60: time.sleep(5)          # 1) 이전 작업 프로세스 종료 대기
    if my_procs(): log(f"{a.tag}: 경고 - 이전 작업 프로세스가 아직 있음: {my_procs()}")
    n = evict(a.paths)                                                             # 2) 이전 작업이 읽은 파일 캐시 비우기
    t1 = time.time()
    while avail_gb() < a.min_avail_gb and time.time() - t1 < a.wait_min * 60: time.sleep(10)   # 3) 가용 메모리 회복 대기
    g = gpu_apps()
    log(f"{a.tag}: 끝    가용 {avail_gb():.1f}GB  파일캐시 {cached_gb():.1f}GB  캐시 비운 파일 {n}개  GPU {g or '없음'}"
        + ("" if avail_gb() >= a.min_avail_gb else f"  경고 - 가용 메모리가 {a.min_avail_gb}GB 미만"))
