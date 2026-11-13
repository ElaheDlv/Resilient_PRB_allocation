"""Bounded CPU-only functional training smoke; never a convergence experiment."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'backend/evaluations/phase1/main_scenarios/scenario_j20.json'))
    p.add_argument('--intervals', type=int, default=300)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--seed', type=int, default=3456)
    p.add_argument('--scripted', action='store_true', help='Deterministic action cycle for integration tests')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', required=True)
    a = p.parse_args()
    if not 2 <= a.intervals <= 400 or not 1 <= a.episodes <= 3:
        p.error('Smoke bounds: 2–400 intervals and 1–3 sequential episodes')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) > 4:
        p.error('Run through phase2b_guard.py: <=4 CPU affinity and CUDA disabled are required')
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(var) != '1':
            p.error(f'{var}=1 required for bounded smoke')
    original = json.loads(Path(a.config).read_text())
    spec = original['episodes'][0]
    catalog = {'episodes': [dict(spec, id=f"{spec['id']}_smoke{i}", duration_steps=a.intervals)
                            for i in range(a.episodes)]}
    with tempfile.TemporaryDirectory(prefix='phase2b-') as tmp:
        config_path = Path(tmp)/'catalog.json'
        config_path.write_text(json.dumps(catalog))
        os.environ.update(RAN_TOPOLOGY_PRESET='simple', SIM_FREEZE_MOBILITY='1',
            SIM_STEP_TIME_DEFAULT='0.001', UE_SPAWN_RADIUS_M='400', UE_DEFAULT_MAX_COUNT='100',
            PRB_GYM_CONFIG_PATH=str(config_path), RES_DQN_TRANSITION_MODE='causal',
            RES_DQN_DECISION_PERIOD='10', PRB_UE_SCHEDULER='proportional_wc',
            PHASE0_MEASUREMENTS='1', PHASE0_MEASUREMENT_PATH='',
            JAM_RESERVED_PRBS='18', RES_DQN_TOTAL_RESILIENCY_PRB='18', DEADLINE_STARVATION_TICKS='3')
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        from network_layer.simulation_engine import SimulationEngine
        from network_layer.xApps.xapp_joint_deadline_dqn import xAppJointDeadlineDQN
        from network_layer.xApps.xapp_scripted_joint_ric import default_script
        from network_layer.joint_dqn_schema import encode_action
        from network_layer.joint_dqn_agent import JointDQNAgent, TrainingConfig
        cfg = TrainingConfig(seed=a.seed)
        selector = (lambda state, mask, i: encode_action(default_script(None, None, i))) if a.scripted else None
        engine = SimulationEngine()
        engine.reset_network()
        engine.network_setup(xapp_classes=[partial(xAppJointDeadlineDQN, enabled=True,
            config=cfg, action_selector=selector)])
        app = next(iter(engine.ric.xapp_list.values()))
        started = time.perf_counter()
        for _ in range(a.intervals*10*a.episodes):
            engine.sim_step += 1
            engine.step(0.001)
            if app.finished:
                break
        elapsed = time.perf_counter()-started
        if not app.finished:
            raise RuntimeError('Smoke ended with an unfinished episode')
        result = app.result()
        times = result.pop('boundary_timings_ns')
        trace_hashes = {}
        for path in sorted({v['trace'] for v in spec['slices'].values() if v.get('trace')}):
            with Path(path).open('rb') as trace:
                trace_hashes[path] = hashlib.file_digest(trace, 'sha256').hexdigest()
        run_metadata = dict(effective_catalog=catalog, steps=engine.sim_step,
            catalog_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            trace_sha256=trace_hashes, scripted=a.scripted,
            replay_resume_supported=False)
        app.agent.save(a.checkpoint, run_metadata=run_metadata)
        loaded = JointDQNAgent(cfg, app.reward_config, app.agent.observation_config)
        loaded.load(a.checkpoint)
        if loaded.fingerprint() != app.agent.fingerprint() or loaded.checkpoint_run_metadata != run_metadata:
            raise AssertionError('Checkpoint parameter round-trip mismatch')
        if torch.cuda.is_initialized() or any(p.device.type != 'cpu' for p in app.agent.policy.parameters()):
            raise AssertionError('GPU unexpectedly used')
        result.update(seed=a.seed, training_config=asdict(cfg), reward_config=asdict(app.reward_config),
            checkpoint_metadata=app.agent.metadata(), checkpoint_run_metadata=run_metadata,
            observation_config=dict(window_s=app.window_s, minimum_matured=app.minimum_matured),
            effective_catalog=catalog, catalog_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            steps=engine.sim_step, simulation_seconds=engine.sim_step*0.001, elapsed_seconds=elapsed,
            boundary_median_us=statistics.median(times)/1000,
            boundary_p95_us=sorted(times)[max(0, int(len(times)*0.95)-1)]/1000,
            checkpoint_roundtrip=True, cuda_initialized=torch.cuda.is_initialized(),
            parameters_changed=result['initial_parameter_sha256'] != result['final_parameter_sha256'],
            cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
            dataloader_workers=0, scripted=a.scripted,
            versions=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__))
        app.close()
        for bs in engine.base_station_list.values():
            bs.queue_measurements.close()
        Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in
            {'first_examples', 'last_examples', 'service_tail', 'losses', 'episode_summaries', 'effective_catalog'}}, indent=2))


if __name__ == '__main__':
    main()
