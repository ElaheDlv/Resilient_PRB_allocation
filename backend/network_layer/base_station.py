import time
import random
import settings
from .cell import Cell
from .edge_server import EdgeServer
import logging
import base64
from utils.traffic_trace import estimate_trace_period
from collections import deque
from .queue_measurements import QueueMeasurements



logger = logging.getLogger(__name__)

SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")

class BaseStation:
    def __init__(self, simulation_engine, bs_init_data):
        assert simulation_engine is not None, "Simulation engine cannot be None"
        assert bs_init_data is not None, "Base station init data cannot be None"

        self.simulation_engine = simulation_engine
        self.core_network = simulation_engine.core_network

        self.bs_id = bs_init_data["bs_id"]
        self.position_x = bs_init_data["position_x"]
        self.position_y = bs_init_data["position_y"]
        self.cell_list = {}
        for cell_init_data in bs_init_data["cell_list"]:
            new_cell = Cell(
                base_station=self,
                cell_init_data=cell_init_data,
            )
            self.cell_list[cell_init_data["cell_id"]] = new_cell
        self.rrc_measurement_events = bs_init_data["rrc_measurement_events"]

        self.edge_server = EdgeServer(self, bs_init_data["edge_server"])

        self.ue_registry = {}
        self.ue_rrc_meas_events = []
        self.ue_rrc_meas_event_handers = {}

        self.ric_control_actions = []

        self.ai_service_event_handler = None

        # ---------------- DL replay and buffering (gNB-side) ----------------
        # Per-UE DL buffer in bytes: represents queued downlink traffic waiting at the gNB.
        self._dl_buf = {}  # {imsi: int}
        # Per-UE arrival timestamps for DL bins (FIFO). Each entry: (t_arrival_s, bytes)
        self._dl_buf_queue = {}  # {imsi: deque[(float, int)]}
        # Per-UE last observed queuing delay (seconds) for the bytes served in the latest pull.
        self._dl_last_delay_s = {}  # {imsi: float}
        # Per-UE DL replay state: holds trace samples plus runtime cursor/clock info.
        self._dl_replay = {}  # {imsi: {samples, idx, clock_s, speedup, period_s}}
        # Wall-clock (simulation) time accumulator for stamping arrivals.
        self._time_s = 0.0
        # Jammer hook: number of DL PRBs jammed for this BS's cells (set per-step).
        self._jam_const_prbs = 0
        self.queue_measurements = (
            QueueMeasurements(getattr(settings, "PHASE0_MEASUREMENT_PATH", ""))
            if getattr(settings, "PHASE0_MEASUREMENTS", False) else None
        )

    def __repr__(self):
        return f"BS {self.bs_id}"

    def receive_ue_rrc_meas_events(self, event):
        # sanity check
        ue = event["triggering_ue"]
        current_cell = self.cell_list.get(event["current_cell_id"], None)
        assert ue is not None, "UE cannot be None"
        assert current_cell is not None, "Current cell cannot be None"
        assert (
            ue.current_cell.cell_id == current_cell.cell_id
        ), f"UE {ue.ue_imsi} (current cell: {ue.current_cell.cell_id}) is not in the current cell ({current_cell.cell_id})"
        logger.info(f"{self} received UE reported RRC measurement event:")
        logger.info(event)
        self.ue_rrc_meas_events.append(event)

    # ---------------------- DL buffer / replay APIs ----------------------
    def attach_dl_trace(self, ue_imsi, samples, speedup: float = 1.0):
        """Attach a DL trace (samples list of (t_s, dl_bytes, ul_bytes)) to this BS for a UE.

        Overview of the replay pipeline:
        - Trace samples arrive as (time, dl_bytes, ul_bytes) buckets where *time* marks the
          start of each bin. When replaying we must enqueue bytes at `time*speedup`.
        - The base station maintains a replay clock per UE; each simulator tick advances the
          clock and drains all samples whose (scaled) time has elapsed into the DL buffer.
        - Cells pull bytes from that buffer when they schedule downlink transmissions.
        """
        if not samples:
            return False
        period_hint = getattr(settings, "TRACE_BIN", None)
        period = float(estimate_trace_period(samples, default_step=period_hint))
        print(f"Estimated trace period: {period} s")
        self._dl_replay[ue_imsi] = {
            "samples": list(samples),
            "idx": 0,
            "clock_s": 0.0,
            "speedup": max(1e-6, float(speedup)),
            "period_s": float(period),
        }
        # Ensure buffer entry exists
        self._dl_buf.setdefault(ue_imsi, 0)
        return True

    def enqueue_dl_bytes(self, ue_imsi: str, nbytes: int) -> int:
        if nbytes <= 0:
            return 0
        cur = int(self._dl_buf.get(ue_imsi, 0) or 0)
        limit = max(0, int(getattr(settings, "TRACE_DL_BUFFER_LIMIT_BYTES", 0) or 0))
        add = int(nbytes)
        if limit > 0:
            space = max(0, limit - cur)
            if space <= 0:
                if self.queue_measurements is not None:
                    self.queue_measurements.arrival(ue_imsi, self._time_s, int(nbytes), 0)
                return 0
            if add > space:
                if getattr(settings, "TRACE_DEBUG", False):
                    logger.debug(
                        "[trace] %s DL buffer capped: requested=%d space=%d limit=%d",
                        ue_imsi,
                        nbytes,
                        space,
                        limit,
                    )
                add = space
        self._dl_buf[ue_imsi] = cur + add
        # Track arrival time for queuing delay (per bin treated as a packet batch)
        q = self._dl_buf_queue.setdefault(ue_imsi, deque())
        if add > 0:
            q.append((self._time_s, add))
        if self.queue_measurements is not None:
            self.queue_measurements.arrival(ue_imsi, self._time_s, int(nbytes), add)
        return add

    def pull_dl_bytes(self, ue_imsi: str, cap_bytes: int) -> int:
        cur = int(self._dl_buf.get(ue_imsi, 0) or 0)
        take = max(0, min(cur, int(cap_bytes)))
        self._dl_buf[ue_imsi] = cur - take

        # Drain from FIFO queue to compute queuing delay for the served bytes
        served = 0
        delay_weighted = 0.0
        q = self._dl_buf_queue.get(ue_imsi)
        if q is None:
            q = deque()
            self._dl_buf_queue[ue_imsi] = q
        remaining = take
        while q and remaining > 0:
            t_arrival, b = q[0]
            use = min(b, remaining)
            served += use
            delay_weighted += use * max(0.0, self._time_s - float(t_arrival))
            b_after = b - use
            remaining -= use
            if b_after > 0:
                q[0] = (t_arrival, b_after)
            else:
                q.popleft()
        # Record bytes-weighted queuing delay (seconds) for this pull
        if served > 0:
            self._dl_last_delay_s[ue_imsi] = delay_weighted / served
        else:
            # Preserve previous delay if nothing was served
            self._dl_last_delay_s.setdefault(ue_imsi, 0.0)

        if self.queue_measurements is not None:
            self.queue_measurements.service(ue_imsi, self._time_s, take)
        return take

    def get_dl_buf_bytes(self, ue_imsi: str) -> int:
        return int(self._dl_buf.get(ue_imsi, 0) or 0)

    def get_dl_last_delay_s(self, ue_imsi: str) -> float:
        return float(self._dl_last_delay_s.get(ue_imsi, 0.0))

    def get_dl_head_wait_s(self, ue_imsi: str) -> float:
        """Time the oldest queued bytes have been waiting (t_now - t_arrival)."""
        q = self._dl_buf_queue.get(ue_imsi)
        if not q:
            return 0.0
        try:
            t_arrival, _ = q[0]
            return max(0.0, float(self._time_s) - float(t_arrival))
        except Exception:
            return 0.0

    def has_dl_replay(self, ue_imsi: str) -> bool:
        return ue_imsi in self._dl_replay

    def tick_dl_replay(self, dt: float):
        """Advance all UE DL replay clocks and enqueue any due arrivals into the gNB buffer."""
        try:
            step = float(dt)
        except Exception:
            step = 0.0
        if step <= 0:
            return
        loop_enabled = bool(getattr(settings, "TRACE_LOOP", False))
        for imsi, st in list(self._dl_replay.items()):
            samples = st.get("samples") or []
            if not samples:
                continue
            spd = float(st.get("speedup", 1.0) or 1.0)
            # Advance replay clock by dt*speedup so faster replays compress time.
            st["clock_s"] = float(st.get("clock_s", 0.0)) + step * spd
            # Round clock to limit floating drift so bins are not skipped at ~1ms scales.
            clock = round(st["clock_s"], 9)
            st["clock_s"] = clock
            # Use a small negative guard so bins landing exactly on the boundary are picked
            # up in the *next* tick (avoids double consumption at 1 kHz).
            eps = max(1e-9, abs(step) * 1e-6)
            # idx tracks the next sample to enqueue for this UE.
            idx = int(st.get("idx", 0))
            n = len(samples)
            period = float(st.get("period_s", 0.0) or 0.0)

            while True:
                # Drain all samples whose timestamps are <= current clock. Each sample time is
                # compared against the scaled replay clock so late-arriving bins get enqueued
                # exactly once per cycle.
                # if int(clock) % 100 == 0:
                #     print(f"[trace-ts] {imsi}: sample_t={samples[idx][0]:.6f}s clock={clock:.6f}s idx={idx}")
                # Consume only bins strictly before the current clock (minus epsilon) so
                # bins that align with the tick boundary are consumed in the following
                # step. This prevents "2 then 0" cadence when dt == bin width.
                #print("clock:", clock)
                while idx < n and float(samples[idx][0]) <= clock - eps:
                    dl = int(samples[idx][1] or 0)
                    #print("sample_t:", samples[idx][0], "dl_bytes:", dl)
                    if dl > 0:
                        self.enqueue_dl_bytes(imsi, dl)
                    idx += 1

                st["idx"] = idx

                if not (loop_enabled and period > 0 and idx >= n and clock >= period):
                    break

                # If looping, wrap the clock and continue in the same simulator tick. This
                # preserves bursts that straddle the end/start boundary and ensures the last
                # sample is replayed before the first sample of the next cycle.
                while st["clock_s"] >= period:
                    st["clock_s"] -= period
                clock = st["clock_s"]
                idx = 0

    def handle_ue_authentication_and_registration(self, ue):
        core_response = self.core_network.handle_ue_authentication_and_registration(ue)
        ue_reg_data = {
            "ue": ue,
            "slice_type": core_response["slice_type"],
            "qos_profile": core_response["qos_profile"],
            "cell": ue.current_cell,
            "rrc_meas_events": self.rrc_measurement_events.copy(),
        }
        self.ue_registry[ue.ue_imsi] = ue_reg_data
        ue.current_cell.register_ue(ue)
        # Ensure gNB DL buffer entry exists for this UE (even if no trace yet)
        try:
            if hasattr(self, "_dl_buf"):
                self._dl_buf.setdefault(ue.ue_imsi, 0)
        except Exception:
            pass
        return ue_reg_data.copy()

    def handle_deregistration_request(self, ue):
        if self.queue_measurements is not None:
            self.queue_measurements.finalize(ue.ue_imsi, self._time_s)
        self.core_network.handle_deregistration_request(ue)
        # Release radio resources where the UE was anchored (if any).
        cell = getattr(ue, "current_cell", None)
        if cell is not None:
            cell.deregister_ue(ue)
        # Drop UE-specific DL buffer and trace replay state so future episodes start clean.
        try:
            if hasattr(self, "_dl_buf"):
                self._dl_buf.pop(ue.ue_imsi, None)
            if hasattr(self, "_dl_replay"):
                self._dl_replay.pop(ue.ue_imsi, None)
        except Exception:
            pass
        if ue.ue_imsi in self.ue_registry:
            del self.ue_registry[ue.ue_imsi]
        # Drop per-UE DL queue state
        try:
            if hasattr(self, "_dl_buf_queue"):
                self._dl_buf_queue.pop(ue.ue_imsi, None)
            if hasattr(self, "_dl_last_delay_s"):
                self._dl_last_delay_s.pop(ue.ue_imsi, None)
        except Exception:
            pass

        # remove rrc measurement events for the UE
        events_to_remove = []
        for event in self.ue_rrc_meas_events:
            if event["triggering_ue"] == ue:
                events_to_remove.append(event)
        for event in events_to_remove:
            self.ue_rrc_meas_events.remove(event)

        logger.info(
            f"gNB {self.bs_id}: UE {ue.ue_imsi} deregistered and resources released."
        )
        return True

    def to_json(self):
        return {
            "bs_id": self.bs_id,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "vis_position_x": self.position_x * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "vis_position_y": self.position_y * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "ue_registry": list(self.ue_registry.keys()),
            "cell_list": [cell.to_json() for cell in self.cell_list.values()],
            "edge_server": self.edge_server.to_json(),
        }

    def init_rrc_measurement_event_handler(self, event_id, handler):
        assert event_id is not None, "Event ID cannot be None"
        assert handler is not None, "Handler cannot be None"
        assert (
            event_id not in self.ue_rrc_meas_event_handers
        ), f"Handler for event ID {event_id} already registered"
        self.ue_rrc_meas_event_handers[event_id] = handler

    def init_ai_service_event_handler(self, handler):
        assert handler is not None, "Handler cannot be None"
        self.ai_service_event_handler = handler

    def process_ric_control_actions(self):
        # only handover actions are supported for now

        # check if there are multiple handover actions for the same UE,
        # reject or merge wherever necessary
        ue_handover_actions = {}
        for action in self.ric_control_actions:
            if action.action_type != action.ACTION_TYPE_HANDOVER:
                logger.info(
                    f"gNB {self.bs_id}: Ignoring non-handover action: {action.action_type}"
                )
                continue

            ue = action.action_data["ue"]
            if ue.ue_imsi not in ue_handover_actions:
                ue_handover_actions[ue.ue_imsi] = []
            ue_handover_actions[ue.ue_imsi].append(action)

        # process each UE's handover actions
        for ue_imsi, actions in ue_handover_actions.items():
            # for now perform the first handover action only.
            action = actions[0]
            ue = action.action_data["ue"]
            source_cell_id = action.action_data["source_cell_id"]
            target_cell_id = action.action_data["target_cell_id"]
            source_cell = self.simulation_engine.cell_list[source_cell_id]
            target_cell = self.simulation_engine.cell_list[target_cell_id]
            self.execute_handover(ue, source_cell, target_cell)
            break

    def execute_handover(self, ue, source_cell, target_cell):
        assert ue is not None, "UE cannot be None"
        assert (
            source_cell is not None and target_cell is not None
        ), "Source or target cell cannot be None"
        assert source_cell != target_cell, "Source and target cell cannot be the same"
        assert (
            ue.current_cell.cell_id == source_cell.cell_id
        ), f"UE {ue.ue_imsi} (current cell: {ue.current_cell.cell_id})is not in the source cell ({source_cell.cell_id})"
        assert (
            ue.ue_imsi in source_cell.connected_ue_list
        ), "UE is not connected to the source cell"
        assert (
            ue.ue_imsi not in target_cell.connected_ue_list
        ), "UE is already connected to the target cell"

        source_bs = source_cell.base_station
        target_bs = target_cell.base_station

        if source_bs.bs_id == target_bs.bs_id:
            # same base station, just change the cell
            target_cell.register_ue(ue)
            ue.execute_handover(target_cell)
            self.ue_registry[ue.ue_imsi]["cell"] = target_cell
            source_cell.deregister_ue(ue)
            logger.info(
                f"gNB {self.bs_id}: Handover UE {ue.ue_imsi} from cell {source_cell.cell_id} to cell {target_cell.cell_id}"
            )
        else:
            ue_reg_data = source_bs.ue_registry[ue.ue_imsi].copy()
            ue_reg_data["cell"] = target_cell
            ue_reg_data["rrc_meas_events"] = target_bs.rrc_measurement_events.copy()
            target_bs.ue_registry[ue.ue_imsi] = ue_reg_data
            target_cell.register_ue(ue)
            # Move DL buffer + replay state across BSs (if present)
            try:
                buf_bytes = 0
                if hasattr(source_bs, "_dl_buf") and ue.ue_imsi in getattr(source_bs, "_dl_buf", {}):
                    buf_bytes = int(source_bs._dl_buf.pop(ue.ue_imsi, 0) or 0)
                buf_q = None
                if hasattr(source_bs, "_dl_buf_queue"):
                    buf_q = source_bs._dl_buf_queue.pop(ue.ue_imsi, None)
                last_delay = None
                if hasattr(source_bs, "_dl_last_delay_s"):
                    last_delay = source_bs._dl_last_delay_s.pop(ue.ue_imsi, None)
                st = None
                if hasattr(source_bs, "_dl_replay") and ue.ue_imsi in getattr(source_bs, "_dl_replay", {}):
                    st = source_bs._dl_replay.pop(ue.ue_imsi, None)
                if hasattr(target_bs, "_dl_buf"):
                    target_bs._dl_buf[ue.ue_imsi] = int(getattr(target_bs, "_dl_buf", {}).get(ue.ue_imsi, 0) or 0) + buf_bytes
                if buf_q is not None and hasattr(target_bs, "_dl_buf_queue"):
                    target_bs._dl_buf_queue[ue.ue_imsi] = buf_q
                if last_delay is not None and hasattr(target_bs, "_dl_last_delay_s"):
                    target_bs._dl_last_delay_s[ue.ue_imsi] = last_delay
                if st is not None and hasattr(target_bs, "_dl_replay"):
                    target_bs._dl_replay[ue.ue_imsi] = st
            except Exception:
                pass
            ue.execute_handover(target_cell)
            source_cell.deregister_ue(ue)
            del source_bs.ue_registry[ue.ue_imsi]
            logger.info(
                f"gNB {self.bs_id} Handover UE {ue.ue_imsi} from cell {source_cell.cell_id} to BS: {target_bs.bs_id} cell {target_cell.cell_id} (different BS)"
            )

    def step(self, delta_time):
        # Advance local simulation time for arrival/queuing stamps
        try:
            self._time_s += float(delta_time)
        except Exception:
            pass
        # Apply optional jamming window (sample jammer for testing)
        jam_var_period = int(getattr(settings, "JAM_VAR_PERIOD_STEPS", -1))
        jam_var_powers = list(getattr(settings, "JAM_VAR_PRBS_LIST", []) or [])
        jam_var_duty = list(getattr(settings, "JAM_VAR_DUTY_LIST", []) or [])
        if jam_var_period > 0 and jam_var_powers and jam_var_duty:
            self.apply_variable_jamming_from_settings()
        else:
            self._apply_constant_jamming()
        # Tick DL replayers (enqueue arrivals) before cells serve
        try:
            self.tick_dl_replay(delta_time)
        except Exception:
            pass
        # then update cells (serving, scheduling)
        for cell in self.cell_list.values():
            cell.step(delta_time)
            if self.queue_measurements is not None:
                for imsi in cell.connected_ue_list:
                    self.queue_measurements.sample(imsi, self._time_s)

        # reset RIC control actions
        self.ric_control_actions = []

        # process RRC measurement events
        while len(self.ue_rrc_meas_events) > 0:
            event = self.ue_rrc_meas_events.pop(0)
            event_id = event["event_id"]
            if event_id not in self.ue_rrc_meas_event_handers:
                logger.info(
                    f"gNB {self.bs_id}: No handler for event ID {event_id}. Skipping."
                )
                continue
            handler = self.ue_rrc_meas_event_handers[event_id]
            action = handler(event)

            if action is not None:
                # add the action to the RIC control actions list
                self.ric_control_actions.append(action)

            logger.info(
                f"gNB {self.bs_id}: Processed RRC measurement event {event_id} for UE {event["triggering_ue"].ue_imsi}"
            )

        # process (reject, merge or execute) all the RIC control actions
        self.process_ric_control_actions()

    def _apply_constant_jamming(self):
        """Sample jammer: periodically remove DL PRBs during configured step windows.

        Controls (settings.py):
            JAM_CONST_PERIOD_STEPS: how often to start a jam window (e.g., 10)
            JAM_CONST_DURATION_STEPS: how many steps the jam lasts once started (e.g., 10)
            JAM_CONST_PRBS: number of DL PRBs to jam during the window (e.g., 5)
            JAM_CONST_START_OFFSET: optional initial offset before the first jam window (default 0)
        """
        sim_step = getattr(self.simulation_engine, "sim_step", 0)
        period = int(getattr(settings, "JAM_CONST_PERIOD_STEPS", -1))
        duration = max(0, int(getattr(settings, "JAM_CONST_DURATION_STEPS", 0)))
        jam_prbs = max(0, int(getattr(settings, "JAM_CONST_PRBS", 0)))
        offset = max(0, int(getattr(settings, "JAM_CONST_START_OFFSET", 0)))
        # Active if we're in the duration window after each period tick, post-offset.
        active = False
        if period > 0 and jam_prbs > 0 and duration > 0 and sim_step >= offset:
            rel = sim_step - offset
            pos_in_period = rel % period
            active = pos_in_period < duration
        reserved = max(0, int(getattr(settings, "JAM_RESERVED_PRBS", 0)))
        for cell in self.cell_list.values():
            # Persist the initial per-slice quotas as the jammer's reference (set once).
            if not hasattr(cell, "_jam_base_quota"):
                base_snapshot = dict(cell.slice_dl_prb_quota)
                cell._jam_base_quota = base_snapshot
                cell.base_dl_prb_quota = dict(base_snapshot)
                cell._jam_was_active = False
            elif active and not getattr(cell, "_jam_was_active", False):
                # Refresh the jammer baseline at the start of a jam window so episode-level
                # resets don't reuse stale quotas from previous runs.
                cur_snapshot = dict(cell.slice_dl_prb_quota)
                # if cur_snapshot and cur_snapshot != getattr(cell, "_jam_base_quota", {}):
                #     print("this is a new jam snapshot:", cur_snapshot)
                #     cell._jam_base_quota = dict(cur_snapshot)
                #     cell.base_dl_prb_quota = dict(cur_snapshot)
            prev_base_quota = dict(getattr(cell, "base_dl_prb_quota", getattr(cell, "slice_dl_prb_quota", {})) or {})
            # Base usable pool excludes reserved PRBs; jam subtracts from the usable pool only.
            usable_max = max(0, int(cell.max_dl_prb) - reserved)
            jam_active = jam_prbs if active else 0
            target_max = max(0, usable_max - jam_active)
            # Scale the jammer's baseline quotas down proportionally if jammed capacity is smaller.
            base_quota = dict(getattr(cell, "_jam_base_quota", getattr(cell, "slice_dl_prb_quota", {})) or {})
            total = sum(base_quota.values())
            # base_e = float(base_quota.get(SL_E, 0) or 0.0)
            # base_u = float(base_quota.get(SL_U, 0) or 0.0)
            #print("base_e:", base_e, "base_u:", base_u)
            quotas_curr = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            # cur_e = float(quotas_curr.get(SL_E, 0) or 0)
            # cur_u = float(quotas_curr.get(SL_U, 0) or 0)
            def _rebuild_live_quota(new_base):
                """Keep reserved extras intact while shrinking/expanding the base quota."""
                extras = {}
                extra_total = 0
                for s in set(quotas_curr.keys()) | set(prev_base_quota.keys()):
                    cur_val = float(quotas_curr.get(s, 0) or 0.0)
                    prev_base = float(prev_base_quota.get(s, 0) or 0.0)
                    extra = max(0.0, cur_val - prev_base)
                    extras[s] = extra
                    extra_total += extra
                # Extra capacity cannot exceed the reserved pool; scale down if necessary.
                extra_budget = min(float(reserved), extra_total)
                scaled_extras = {s: 0 for s in extras.keys()}
                if extra_total > 0 and extra_budget > 0:
                    if extra_total <= extra_budget:
                        for s, v in extras.items():
                            scaled_extras[s] = int(round(v))
                    else:
                        scale_extra = extra_budget / float(extra_total)
                        remainders = []
                        for s, v in extras.items():
                            raw = v * scale_extra
                            down = int(raw)
                            scaled_extras[s] = down
                            remainders.append((raw - down, s))
                        leftover_extra = int(round(extra_budget - sum(scaled_extras.values())))
                        remainders.sort(reverse=True)
                        for _, s in remainders:
                            if leftover_extra <= 0:
                                break
                            scaled_extras[s] += 1
                            leftover_extra -= 1
                # Update live quotas to the jammed base plus reserved extras.
                new_quota = {}
                for s in set(new_base.keys()) | set(scaled_extras.keys()):
                    base_val = int(new_base.get(s, 0) or 0)
                    extra_val = int(scaled_extras.get(s, 0) or 0)
                    new_quota[s] = base_val + extra_val
                cell.slice_dl_prb_quota = new_quota
                
                
                
            if total > 0 and target_max < total:
                scale = target_max / float(total)
                # Scale proportionally and distribute any remainder to match target_max.
                scaled = {}
                remainders = []
                for s, v in base_quota.items():
                    raw = v * scale
                    down = int(raw)
                    scaled[s] = down
                    remainders.append((raw - down, s))
                leftover = target_max - sum(scaled.values())
                # Give leftover PRBs to slices with largest fractional remainder.
                remainders.sort(reverse=True)
                for _, s in remainders:
                    if leftover <= 0:
                        break
                    scaled[s] += 1
                    leftover -= 1
                base_quota = scaled
                cell.base_dl_prb_quota = base_quota
                cell._jam_was_active = True
                _rebuild_live_quota(base_quota)
                # base_e_new = float(base_quota.get(SL_E, 0) or 0.0)
                # base_u_new = float(base_quota.get(SL_U, 0) or 0.0)
                # #print("base_e_new:", base_e_new, "base_u_new:", base_u_new)

                # if base_e_new < cur_e or base_u_new < cur_u:
                #     cur_e_delta = cur_e - base_e
                #     cur_u_delta = cur_u - base_u
                #     print("cur_e_delta:", cur_e_delta, "cur_u_delta:", cur_u_delta)
                #     # adjust current quotas based on the new base quotas
                #     # base_e_new = float(base_quota.get(SL_E, 0) or 0.0)
                #     # base_u_new = float(base_quota.get(SL_U, 0) or 0.0)
                #     # print("base_e_new:", base_e_new, "base_u_new:", base_u_new)
                
                #     cur_e_new = base_e_new + cur_e_delta
                #     cur_u_new = base_u_new + cur_u_delta
                #     print("cur_e_new:", cur_e_new, "cur_u_new:", cur_u_new)

                #     quotas = {SL_E: int(round(cur_e_new)), SL_U: int(round(cur_u_new))}
                #     cell.slice_dl_prb_quota = quotas
                
                


            # ### code for change of the slice dl prb quota during jamming
            # dl_quota = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            # total_dl = sum(dl_quota.values())
            # if total_dl > 0 and target_max < total_dl:
            #     scale_dl = target_max / float(total_dl)
            #     # Scale proportionally and distribute any remainder to match target_max.
            #     scaled_dl = {}
            #     remainders_dl = []
            #     for s, v in dl_quota.items():
            #         raw_dl = v * scale_dl
            #         down_dl = int(raw_dl)
            #         scaled_dl[s] = down_dl
            #         remainders_dl.append((raw_dl - down_dl, s))
            #     leftover_dl = target_max - sum(scaled_dl.values())
            #     # Give leftover PRBs to slices with largest fractional remainder.
            #     remainders_dl.sort(reverse=True)
            #     for _, s in remainders_dl:
            #         if leftover_dl <= 0:
            #             break
            #         scaled_dl[s] += 1
            #         leftover_dl -= 1
            #     dl_quota = scaled_dl
            #     cell.old_max = target_max
            #     cell.slice_dl_prb_quota = dl_quota
                #print(f"Jammer scaling cell {cell.cell_id} quotas to {base_quota}")
                # Align live quotas to the jammed baseline once; allocator will adjust thereafter.
                #cell.slice_dl_prb_quota = dict(base_quota)
                #cell._jam_was_active = True
            elif not active and getattr(cell, "_jam_was_active", False):
                # Restore base quotas when coming out of a jam window.
                cell.base_dl_prb_quota = dict(base_quota)
                print('restoring base quota:', base_quota)
                
                
                # if base_e>=cur_e and base_u>=cur_u:
                #     cur_e_delta = base_e - cur_e 
                #     cur_u_delta = base_u - cur_u
                # # adjust current quotas based on the new base quotas
                # base_e_new = float(base_quota.get(SL_E, 0) or 0.0)
                # base_u_new = float(base_quota.get(SL_U, 0) or 0.0)
                
                # cur_e_new = base_e_new + cur_e_delta
                # cur_u_new = base_u_new + cur_u_delta

                # quotas = {SL_E: int(round(cur_e_new)), SL_U: int(round(cur_u_new))}
                # cell.slice_dl_prb_quota = quotas
                
                #cell.slice_dl_prb_quota = dict(base_quota)
                cell._jam_was_active = False
                _rebuild_live_quota(base_quota)
            #     ### restore the slice dl prb quota after jamming
            # elif not active and getattr(cell, "_jam_was_active", True):
            #     """
            #     Approximate inverse of proportional down-scaling.
            #     Preserves current slice ratios.
            #     """
            #     if total_dl == 0 or cell.old_max == 0:
            #         return dict(dl_quota)

            #     scale_up = usable_max / float(cell.old_max)

            #     restored = {}
            #     remainders = []

            #     for s, v in dl_quota.items():
            #         raw = v * scale_up
            #         down = int(raw)
            #         restored[s] = down
            #         remainders.append((raw - down, s))

            #         leftover = usable_max - sum(restored.values())
            #         remainders.sort(reverse=True)

            #     for _, s in remainders:
            #         if leftover <= 0:
            #             break
            #         restored[s] += 1
            #         leftover -= 1
            #     cell.slice_dl_prb_quota = restored
            #     cell._jam_was_active = False

            # Do not overwrite live quotas; just reduce effective capacity.
            #cell.jammed_dl_prb = reserved + jam_active
            cell._effective_max_dl_prb = target_max
            
        #print(f"Jamming active: {active}")
        #print("base quota:", cell.base_dl_prb_quota)

    def apply_variable_jamming(
        self,
        power_list,
        duty_list,
        period_steps,
        power_policy="random",
        duty_policy="per_episode",
        episode_steps=0,
        start_offset=0,
        rng=None,
    ):
        """Variable jammer with per-window power and per-episode duty cycle."""
        if rng is None:
            rng = random
        try:
            period = int(period_steps)
        except Exception:
            return False
        if period <= 0 or not power_list or not duty_list:
            return False

        power_list = [int(x) for x in power_list]
        duty_list = [float(x) for x in duty_list]

        sim_step = int(getattr(self.simulation_engine, "sim_step", 0))
        offset = max(0, int(start_offset))
        before_offset = sim_step < offset
        rel = sim_step - offset
        if rel < 0:
            rel = 0
        period_idx = rel // period
        pos_in_period = rel % period

        if episode_steps and int(episode_steps) > 0:
            episode_idx = rel // int(episode_steps)
        else:
            episode_idx = 0

        state = getattr(self, "_jam_var_state", None)
        if state is None:
            state = {
                "power_idx": 0,
                "duty_idx": 0,
                "last_period_idx": None,
                "last_episode_idx": None,
                "current_power": None,
                "current_duty": None,
            }
            setattr(self, "_jam_var_state", state)

        power_policy = str(power_policy).lower()
        duty_policy = str(duty_policy).lower()

        if state.get("last_period_idx") != period_idx:
            state["last_period_idx"] = period_idx
            if power_policy == "increasing":
                state["power_idx"] = (state.get("power_idx", 0) + 1) % len(power_list)
            elif power_policy == "decreasing":
                state["power_idx"] = (state.get("power_idx", 0) - 1) % len(power_list)
            elif power_policy == "cycle":
                state["power_idx"] = int(period_idx % len(power_list))
            elif power_policy == "fixed":
                state["power_idx"] = 0
            else:
                state["power_idx"] = rng.randint(0, len(power_list) - 1)
            state["current_power"] = power_list[state["power_idx"]]

        if state.get("last_episode_idx") != episode_idx:
            state["last_episode_idx"] = episode_idx
            if duty_policy == "cycle_per_episode":
                state["duty_idx"] = int(episode_idx % len(duty_list))
            elif duty_policy == "fixed":
                state["duty_idx"] = 0
            else:
                state["duty_idx"] = rng.randint(0, len(duty_list) - 1)
            state["current_duty"] = duty_list[state["duty_idx"]]

        duty = float(state.get("current_duty", duty_list[0]) or 0.0)
        duty = max(0.0, min(1.0, duty))
        duration = int(round(period * duty))
        active = (not before_offset) and duration > 0 and pos_in_period < duration
        jam_prbs = int(state.get("current_power", power_list[0]) or 0)

        reserved = max(0, int(getattr(settings, "JAM_RESERVED_PRBS", 0)))
        for cell in self.cell_list.values():
            # Persist the initial per-slice quotas as the jammer's reference (set once).
            if not hasattr(cell, "_jam_base_quota"):
                base_snapshot = dict(cell.slice_dl_prb_quota)
                cell._jam_base_quota = base_snapshot
                cell.base_dl_prb_quota = dict(base_snapshot)
                cell._jam_was_active = False
            elif active and not getattr(cell, "_jam_was_active", False):
                cur_snapshot = dict(cell.slice_dl_prb_quota)
                # if cur_snapshot and cur_snapshot != getattr(cell, "_jam_base_quota", {}):
                #     cell._jam_base_quota = dict(cur_snapshot)
                #     cell.base_dl_prb_quota = dict(cur_snapshot)
            prev_base_quota = dict(getattr(cell, "base_dl_prb_quota", getattr(cell, "slice_dl_prb_quota", {})) or {})
            usable_max = max(0, int(cell.max_dl_prb) - reserved)
            jam_active = jam_prbs if active else 0
            target_max = max(0, usable_max - jam_active)
            base_quota = dict(getattr(cell, "_jam_base_quota", getattr(cell, "slice_dl_prb_quota", {})) or {})
            total = sum(base_quota.values())
            quotas_curr = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            def _rebuild_live_quota(new_base):
                extras = {}
                extra_total = 0
                for s in set(quotas_curr.keys()) | set(prev_base_quota.keys()):
                    cur_val = float(quotas_curr.get(s, 0) or 0.0)
                    prev_base = float(prev_base_quota.get(s, 0) or 0.0)
                    extra = max(0.0, cur_val - prev_base)
                    extras[s] = extra
                    extra_total += extra
                extra_budget = min(float(reserved), extra_total)
                scaled_extras = {s: 0 for s in extras.keys()}
                if extra_total > 0 and extra_budget > 0:
                    if extra_total <= extra_budget:
                        for s, v in extras.items():
                            scaled_extras[s] = int(round(v))
                    else:
                        scale_extra = extra_budget / float(extra_total)
                        remainders = []
                        for s, v in extras.items():
                            raw = v * scale_extra
                            down = int(raw)
                            scaled_extras[s] = down
                            remainders.append((raw - down, s))
                        leftover_extra = int(round(extra_budget - sum(scaled_extras.values())))
                        remainders.sort(reverse=True)
                        for _, s in remainders:
                            if leftover_extra <= 0:
                                break
                            scaled_extras[s] += 1
                            leftover_extra -= 1
                new_quota = {}
                for s in set(new_base.keys()) | set(scaled_extras.keys()):
                    base_val = int(new_base.get(s, 0) or 0)
                    extra_val = int(scaled_extras.get(s, 0) or 0)
                    new_quota[s] = base_val + extra_val
                cell.slice_dl_prb_quota = new_quota

            if total > 0 and target_max < total:
                scale = target_max / float(total)
                scaled = {}
                remainders = []
                for s, v in base_quota.items():
                    raw = v * scale
                    down = int(raw)
                    scaled[s] = down
                    remainders.append((raw - down, s))
                leftover = target_max - sum(scaled.values())
                remainders.sort(reverse=True)
                for _, s in remainders:
                    if leftover <= 0:
                        break
                    scaled[s] += 1
                    leftover -= 1
                base_quota = scaled
                cell.base_dl_prb_quota = base_quota
                cell._jam_was_active = True
                _rebuild_live_quota(base_quota)
            elif not active and getattr(cell, "_jam_was_active", False):
                cell.base_dl_prb_quota = dict(base_quota)
                cell._jam_was_active = False
                _rebuild_live_quota(base_quota)
            cell._effective_max_dl_prb = target_max
        return True

    def apply_variable_jamming_from_settings(self, rng=None):
        """Apply variable jamming using settings-based parameters."""
        power_list = list(getattr(settings, "JAM_VAR_PRBS_LIST", []) or [])
        duty_list = list(getattr(settings, "JAM_VAR_DUTY_LIST", []) or [])
        period_steps = int(getattr(settings, "JAM_VAR_PERIOD_STEPS", -1))
        power_policy = str(getattr(settings, "JAM_VAR_POWER_POLICY", "random"))
        duty_policy = str(getattr(settings, "JAM_VAR_DUTY_POLICY", "per_episode"))
        episode_steps = int(getattr(settings, "JAM_VAR_EPISODE_STEPS", 0))
        start_offset = int(getattr(settings, "JAM_VAR_START_OFFSET", 0))
        return self.apply_variable_jamming(
            power_list=power_list,
            duty_list=duty_list,
            period_steps=period_steps,
            power_policy=power_policy,
            duty_policy=duty_policy,
            episode_steps=episode_steps,
            start_offset=start_offset,
            rng=rng,
        )

    def on_ue_application_traffic(self, ue, traffic_data):
        url = traffic_data["url"]

        # for the moment, we only support the AI service traffic
        if not url.startswith("http://cranfield_6G.com/ai_services/"):
            return

        # when a connected UE requests edge AI service
        ai_service_name = url.replace("http://cranfield_6G.com/ai_services/", "")
        ue_imsi = traffic_data["data"]["ue_id"]
        if not ai_service_name:
            logger.warning("Undefined ai_service_name")
            return

        # local breakout
        ai_service_subscription = self.edge_server.check_ue_subscription(
            ai_service_name, ue_imsi
        )
        if not ai_service_subscription:
            return

        # forward the request to the edge server
        start_time = time.time() * 1000  # convert to milliseconds
        response = self.edge_server.handle_ai_service_request(
            ai_service_subscription=ai_service_subscription,
            request_data=traffic_data["data"],
            request_files=traffic_data.get("files", {}),
        )

        end_time = time.time() * 1000  # convert to milliseconds

        if self.ai_service_event_handler:
            files = traffic_data.get("files", {})
            request_files_size = 0
            if files and files.get("file", None):
                request_files_size = len(files["file"])
                # encode the files from bytes to base64 string
                files["file_base64"] = base64.b64encode(files["file"]).decode("utf-8")
                del files["file"]
            self.ai_service_event_handler(
                {
                    "ue_imsi": ue.ue_imsi,
                    "request": {
                        "ai_service_name": ai_service_name,
                        "ue_imsi": ue_imsi,
                        "request_data": traffic_data["data"],
                        "request_files": files,
                        "request_files_size": request_files_size,
                    },
                    "response": response,
                    "service_response_time_ms": end_time - start_time,
                }
            )

        return response
