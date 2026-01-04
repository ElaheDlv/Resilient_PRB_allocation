import time
import settings
from .cell import Cell
from .edge_server import EdgeServer
import logging
import base64
from utils.traffic_trace import estimate_trace_period
from collections import deque



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
        self._apply_constant_jamming()
        # Tick DL replayers (enqueue arrivals) before cells serve
        try:
            self.tick_dl_replay(delta_time)
        except Exception:
            pass
        # then update cells (serving, scheduling)
        for cell in self.cell_list.values():
            cell.step(delta_time)

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
            if not hasattr(cell, "base_dl_prb_quota"):
                cell.base_dl_prb_quota = dict(cell.slice_dl_prb_quota)
            # Base usable pool excludes reserved PRBs; jam subtracts from the usable pool only.
            usable_max = max(0, int(cell.max_dl_prb) - reserved)
            jam_active = jam_prbs if active else 0
            target_max = max(0, usable_max - jam_active)
            # Scale the jammer's baseline quotas down proportionally if jammed capacity is smaller.
            base_quota = dict(getattr(cell, "base_dl_prb_quota", {}) or {})
            total = sum(base_quota.values())
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
                # Align live quotas to the jammed baseline once; allocator will adjust thereafter.
                cell.slice_dl_prb_quota = dict(base_quota)
            # Do not overwrite live quotas; just reduce effective capacity.
            cell.jammed_dl_prb = reserved + jam_active
            cell._effective_max_dl_prb = target_max
            
        #print(f"Jamming active: {active}")

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
