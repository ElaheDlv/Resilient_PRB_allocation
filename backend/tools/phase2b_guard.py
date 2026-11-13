"""Run one sequential CPU-only workload with affinity and aggregate RSS monitoring.

No hard cgroup memory limit is claimed. Only the launched process group is stopped.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def processes(group):
    """Read the launched group and its descendants, including nested child sessions."""
    table = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            table.append((int(path.name), int(fields[1]), int(fields[2]),
                          int(fields[11])+int(fields[12]),
                          int(fields[21])*os.sysconf('SC_PAGE_SIZE')))
        except (OSError, ValueError, IndexError):
            continue
    owned = {group} | {pid for pid,_,pg,_,_ in table if pg == group}
    while True:
        descendants = {pid for pid,parent,_,_,_ in table if parent in owned}
        expanded = owned | descendants
        if expanded == owned:
            break
        owned = expanded
    return [(pid,ticks,rss) for pid,_,_,ticks,rss in table if pid in owned]



def stop_workload(root):
    """Stop only tracked descendants, including workers starting a new session."""
    rows = processes(root)
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        handles = []
        for pid, _, _ in rows:
            try:
                handles.append(os.pidfd_open(pid))
            except ProcessLookupError:
                pass
        for handle in reversed(handles):
            try:
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(handle)
    else:
        # Some Python builds omit pidfd support. Only our freshly enumerated
        # descendants are signaled; unrelated process groups are never selected.
        for pid, _, _ in reversed(rows):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        os.killpg(root, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpus', default='8,9,10,15')
    p.add_argument('--report', required=True)
    p.add_argument('--warning-gib', type=float, default=2.0)
    p.add_argument('--stop-gib', type=float, default=3.0)
    p.add_argument('--minimum-available-gib', type=float, default=4.0)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cpus = {int(c) for c in a.cpus.split(',')}
    if not 1 <= len(cpus) <= 4 or not cpus <= os.sched_getaffinity(0):
        p.error('Choose 1–4 available logical CPUs')
    if not 0 < a.warning_gib < a.stop_gib <= 3 or a.minimum_available_gib < 4:
        p.error('Require 0 < warning < stop <= 3 GiB and available threshold >=4 GiB')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('A workload command is required')
    os.sched_setaffinity(0, cpus)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED='0',
               PHASE_GUARD_STOP_BYTES=str(int(a.stop_gib*1024**3)),
               **{v: '1' for v in THREAD_VARIABLES})
    def cpu_sample():
        return {line.split()[0]:list(map(int,line.split()[1:9]))
                for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('cpu') and line[3:4].isdigit()}
    before_cpu = cpu_sample()
    time.sleep(0.2)
    after_cpu = cpu_sample()
    utilization = {}
    for cpu in sorted(cpus):
        delta = [b-a for a,b in zip(before_cpu[f'cpu{cpu}'], after_cpu[f'cpu{cpu}'])]
        utilization[str(cpu)] = round(100*(1-(delta[3]+delta[4])/max(1,sum(delta))),2)
    available = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if int(available.split()[1])*1024 < a.minimum_available_gib*1024**3:
        p.error('Available RAM below conservative 4-GiB launch threshold')
    started = time.monotonic()
    child = subprocess.Popen(command, env=env, start_new_session=True)
    peak_rss = peak_cpu = 0.0
    threshold = int(a.stop_gib*1024**3)
    warning_threshold = int(a.warning_gib*1024**3)
    warned = False
    stop_reason = None
    last_cpu = {}
    last_time = started
    stopped = False
    samples = 0
    own_rss = 0
    try:
        while True:
            now = time.monotonic()
            rows = processes(child.pid)
            own = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
            rows.append((os.getpid(),int(own[11])+int(own[12]),0))
            # Include guard resident memory as well, conservatively summing RSS.
            own_rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            rss = sum(r[2] for r in rows) + own_rss
            delta = sum(max(0, ticks-last_cpu.get(pid, ticks)) for pid,ticks,_ in rows)
            cpu = 100 * delta / os.sysconf('SC_CLK_TCK') / max(now-last_time, 1e-9)
            peak_rss = max(peak_rss, rss)
            peak_cpu = max(peak_cpu, cpu)
            last_cpu = {pid: ticks for pid,ticks,_ in rows}
            last_time = now
            samples += 1
            if rss >= warning_threshold and not warned:
                warned = True
                print('Resource warning: aggregate RSS reached warning threshold', file=sys.stderr, flush=True)
            if rss >= threshold:
                stopped = True
                stop_reason = 'aggregate_rss'
            if samples % 20 == 0:
                available_now = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))
                if int(available_now.split()[1])*1024 < a.minimum_available_gib*1024**3:
                    stopped = True
                    stop_reason = 'system_memory_pressure'
            if stopped and child.poll() is None:
                stop_workload(child.pid)
            if child.poll() is not None:
                break
            time.sleep(0.05)
        code = child.wait()
    except BaseException:
        stop_workload(child.pid)
        child.wait()
        raise
    result = dict(command=command, cpus=sorted(cpus),
        affinity_mechanism='sched_setaffinity inherited by subprocesses',
        peak_process_group_cpu_percent=round(peak_cpu, 2),
        peak_aggregate_rss_bytes=int(peak_rss), sampling_period_s=0.05,
        guard_rss_bytes=own_rss, samples=samples,
        assigned_cpu_busy_percent_before=utilization,
        cpu_measurement_note='Sampled launched process tree plus guard; 100% equals one logical CPU. Short-lived processes may be missed.',
        memory_limit_hard_enforced=False, monitored_stop_threshold_bytes=threshold,
        memory_limit_note='No writable delegated cgroup or user systemd bus; conservative RSS watchdog, not a hard cap.',
        numerical_threads_per_process=1, dataloader_workers=0,
        cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'], available_ram_before=available,
        elapsed_seconds=time.monotonic()-started, stopped_for_memory=stopped, warning_threshold_bytes=warning_threshold,
        warning_triggered=warned, stop_reason=stop_reason,
        minimum_available_gib=a.minimum_available_gib,
        exit_code=code)
    Path(a.report).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    sys.exit(code if code >= 0 else 128-code)


if __name__ == '__main__':
    main()
