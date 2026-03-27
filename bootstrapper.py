import math
import numpy as np
from collections import deque, Counter

# ==========================================
# 1. HARMONIC REGIME DETECTOR (Physics Engine)
# ==========================================
INTERVAL_ANGLES = {
    "1": 0, "b2": 180, "2": 120, "b3": 270, "3": 60, "4": 330,
    "#4": 210, "5": 30, "b6": 300, "6": 90, "b7": 240, "7": 150
}
SEMITONE_MAP = {
    "1": 0, "b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, 
    "#4": 6, "5": 7, "b6": 8, "6": 9, "b7": 10, "7": 11
}

class HarmonicRegimeDetector:
    def __init__(self, buffer_ms=300, debounce_ms=150):
        self.buffer_ms = buffer_ms
        self.debounce_ms = debounce_ms
        self.history = deque()
        self.prev_x, self.prev_y = 0.0, 0.0
        self.last_spike_time = -9999 
        self.prev_bass_id = None 

    def process_frame(self, current_time_ms, active_notes):
        self.history.append((current_time_ms, active_notes))
        while self.history and self.history[0][0] < current_time_ms - self.buffer_ms:
            self.history.popleft()

        current_bass_id = None
        if active_notes:
            lowest_note = min(active_notes, key=lambda n: (n[1], SEMITONE_MAP[n[0]]))
            current_bass_id = f"{lowest_note[0]}_{lowest_note[1]}"

        bass_changed = (current_bass_id and self.prev_bass_id and current_bass_id != self.prev_bass_id)
        self.prev_bass_id = current_bass_id

        x_total, y_total, weight_total = 0.0, 0.0, 0.0
        for _, frame_notes in self.history:
            for interval, octave, velocity in frame_notes:
                if velocity <= 0: continue
                weight = velocity / 127.0
                weight_total += weight
                angle_rad = math.radians(INTERVAL_ANGLES[interval])
                x_total += weight * math.cos(angle_rad)
                y_total += weight * math.sin(angle_rad)

        if weight_total == 0:
            self.prev_x, self.prev_y = 0.0, 0.0
            return {"Time (ms)": current_time_ms, "Hue": 0.0, "Sat (%)": 0.0, "V_vec": 0.0, "State": "Silence"}

        x_avg, y_avg = x_total / weight_total, y_total / weight_total
        v_vec = math.sqrt((x_avg - self.prev_x)**2 + (y_avg - self.prev_y)**2) * 100.0
        self.prev_x, self.prev_y = x_avg, y_avg

        final_hue = math.degrees(math.atan2(y_avg, x_avg))
        if final_hue < 0: final_hue += 360
        final_saturation = math.sqrt(x_avg**2 + y_avg**2) * 100.0

        state = "Stable"
        if final_saturation < 30.0:
            state = "Undefined / Gray Void"
        else:
            is_spiking = (v_vec > 15.0) or bass_changed
            if is_spiking and (current_time_ms - self.last_spike_time) >= self.debounce_ms:
                state = "TRANSITION SPIKE!"
                self.last_spike_time = current_time_ms
            elif final_saturation > 75.0 and v_vec < 5.0:
                state = "Regime Locked"

        return {"Time (ms)": current_time_ms, "Hue": round(final_hue, 1), "Sat (%)": round(final_saturation, 1), "V_vec": round(v_vec, 1), "State": state}


# ==========================================
# 2. DYNAMIC METER DETECTOR (MIR Engine)
# ==========================================
class DynamicMeterDetector:
    def __init__(self, window_size_ms=15000):
        self.window_size_ms = window_size_ms

    def analyze_window(self, start_time, keyframes, regime_frames):
        end_time = start_time + self.window_size_ms
        window_kfs = [k for k in keyframes if start_time <= k[0] <= end_time]
        window_regs = [r for r in regime_frames if start_time <= r["Time (ms)"] <= end_time]
        
        if not window_kfs or not window_regs: return None

        # Master Structure
        spikes = [r["Time (ms)"] for r in window_regs if r["State"] == "TRANSITION SPIKE!"]
        if len(spikes) < 2: return None
        avg_regime_length = np.median([spikes[i] - spikes[i-1] for i in range(1, len(spikes))])

        # Impact Clusters
        impacts = [{"time": t, "impact": sum(n[2] for n in notes if len(n)==3)} for t, notes in window_kfs if sum(n[2] for n in notes if len(n)==3) > 0]
        if not impacts: return None
        
        impact_threshold = np.percentile([i["impact"] for i in impacts], 75)
        cluster_times = [i["time"] for i in impacts if i["impact"] >= impact_threshold]
        if len(cluster_times) < 2: return None
        
        valid_cluster_deltas = [cluster_times[i] - cluster_times[i-1] for i in range(1, len(cluster_times)) if (cluster_times[i] - cluster_times[i-1]) > 200]
        if not valid_cluster_deltas: return None
        avg_cluster_length = np.median(valid_cluster_deltas)

        # Subdivisions
        subdivisions_list = []
        for i in range(1, len(cluster_times)):
            onsets_between = len([k for k in window_kfs if cluster_times[i-1] < k[0] < cluster_times[i]])
            subdivisions_list.append(onsets_between + 1)
        most_common_subdiv = Counter(subdivisions_list).most_common(1)[0][0] if subdivisions_list else 1

        # Math
        beats_per_measure_raw = round(avg_regime_length / avg_cluster_length)
        beats_per_measure = beats_per_measure_raw if beats_per_measure_raw in [2, 3, 4, 6] else 4
        
        bottom = 4
        if most_common_subdiv in [3, 6]:
            beats_per_measure *= 3
            bottom = 8

        return {
            "Beats Per Measure": int(beats_per_measure),
            "Meter Bottom": bottom,
            "Tempo AQNTL (ms)": avg_cluster_length,
            "Impact Threshold": impact_threshold
        }


# ==========================================
# 3. REVERSE ECHOLOCATION BOOTSTRAPPER 
# ==========================================
class Anchor:
    def __init__(self, measure, beat, time_ms, state_note=""):
        self.measure, self.beat, self.time_ms, self.state_note = measure, beat, time_ms, state_note 
    def __repr__(self):
        return f"M{self.measure} B{self.beat} @ {self.time_ms:04d}ms | [{self.state_note}]"

class ReverseEcholocationBootstrapper:
    def __init__(self, regime_frames, keyframes):
        self.frames = regime_frames 
        self.keyframes = keyframes
        self.anchors = []
        self.current_measure, self.current_beat = 1, 1
        self.last_anchor_time = 0.0
        self.max_time = float(self.frames[-1]["Time (ms)"]) if self.frames else 0
        
        # Initialize dynamically
        self.meter_detector = DynamicMeterDetector()
        self.beats_per_measure = 4
        self.aqntl = 500.0

    def run(self):
        if not self.frames: return []

        # --- 1. DYNAMIC INITIALIZATION ---
        meter_data = self.meter_detector.analyze_window(0, self.keyframes, self.frames)
        if meter_data:
            self.beats_per_measure = meter_data["Beats Per Measure"]
            self.aqntl = meter_data["Tempo AQNTL (ms)"]
            print(f"Dynamic Meter Locked: {self.beats_per_measure}/{meter_data['Meter Bottom']} at {self.aqntl}ms per beat.")

        # --- 2. BOOTSTRAP: Anacrusis (Pickup) Math ---
        first_spike = self._find_next_spike(0)
        if first_spike is None: return []

        if meter_data:
            # Check the impact of the very first note against our dynamic threshold
            first_note_impact = sum(n[2] for t, notes in self.keyframes if t == first_spike for n in notes if len(n)==3)
            
            if first_note_impact < meter_data["Impact Threshold"]:
                print("Pickup (Anacrusis) detected! Adjusting Downbeat.")
                self.current_measure, self.current_beat = 0, self.beats_per_measure
                self._lock_anchor(first_spike, "Pickup Anchor", update_tempo=False)
                self.current_measure, self.current_beat = 1, 1 # Next beat is M1 B1
            else:
                self._lock_anchor(first_spike, "Seed Anchor (Downbeat)", update_tempo=False)
        else:
            self._lock_anchor(first_spike, "Seed Anchor (Default)", update_tempo=False)

        # --- 3. METRICAL LOOP ---
        while True:
            expected_time = self.last_anchor_time + self.aqntl
            if expected_time > self.max_time: break 

            standard_buffer = self.aqntl * 0.20 
            syncopation_buffer = self.aqntl * 0.50 
            
            spike_time = self._find_spike_in_window(expected_time - standard_buffer, expected_time + standard_buffer)

            # A. On-Beat Transition
            if spike_time is not None:
                self._lock_anchor(spike_time, "On-Beat Transition", update_tempo=True)
                continue
                
            # B. Syncopation Trap
            early_spike = self._find_spike_in_window(expected_time - syncopation_buffer, expected_time - standard_buffer)
            if early_spike is not None:
                self._lock_anchor(early_spike, "Syncopated Anticipation", update_tempo=False)
                self.last_anchor_time = expected_time 
                continue

            state_at_expected = self._get_state_at_time(expected_time)

            # C. Held Chord
            if state_at_expected == "Regime Locked":
                self._lock_anchor(expected_time, "Held Chord (Dead-Reckon)", update_tempo=False)
                continue

            # D. Fermata / Void Math
            if state_at_expected in ["Undefined / Gray Void", "Silence"]:
                post_fermata_spike = self._find_next_spike(expected_time)
                if post_fermata_spike is None: break 
                
                # Re-evaluate meter after a massive break just in case it shifted
                new_meter_data = self.meter_detector.analyze_window(post_fermata_spike, self.keyframes, self.frames)
                if new_meter_data:
                    self.aqntl = new_meter_data["Tempo AQNTL (ms)"]
                    self.beats_per_measure = new_meter_data["Beats Per Measure"]
                
                time_in_void = post_fermata_spike - self.last_anchor_time
                beats_skipped = round(time_in_void / self.aqntl)
                
                if beats_skipped > 1:
                    self._advance_grid_by_beats(beats_skipped - 1)
                    
                self._lock_anchor(post_fermata_spike, "Fermata Resync", update_tempo=False)
                continue

        return self.anchors

    def _lock_anchor(self, time_ms, note, update_tempo=True):
        self.anchors.append(Anchor(self.current_measure, self.current_beat, int(time_ms), state_note=note))
        if update_tempo:
            self.aqntl = (self.aqntl * 0.7) + ((time_ms - self.last_anchor_time) * 0.3) 
        self.last_anchor_time = time_ms
        self._advance_grid_by_beats(1)

    def _advance_grid_by_beats(self, num_beats):
        total_beats = (self.current_beat - 1) + num_beats
        self.current_measure += int(total_beats // self.beats_per_measure)
        self.current_beat = int((total_beats % self.beats_per_measure) + 1)

    def _get_state_at_time(self, target_time):
        return min(self.frames, key=lambda f: abs(f["Time (ms)"] - target_time))["State"]

    def _find_spike_in_window(self, start_ms, end_ms):
        for f in self.frames:
            if start_ms <= f["Time (ms)"] <= end_ms and f["State"] == "TRANSITION SPIKE!": return f["Time (ms)"]
        return None

    def _find_next_spike(self, start_ms):
        for f in self.frames:
            if f["Time (ms)"] > start_ms and f["State"] == "TRANSITION SPIKE!": return f["Time (ms)"]
        return None

# ==========================================
# 4. EXECUTION WRAPPER
# ==========================================
def run_full_pipeline(performance_keyframes, initial_tempo=500.0):
    detector = HarmonicRegimeDetector()
    frames = []
    max_time = performance_keyframes[-1][0] + 500 
    current_idx = 0
    active_notes = []

    for time_ms in range(0, max_time + 10, 10):
        while current_idx < len(performance_keyframes) and performance_keyframes[current_idx][0] <= time_ms:
            active_notes = performance_keyframes[current_idx][1]
            current_idx += 1
        frames.append(detector.process_frame(time_ms, active_notes))

    bootstrapper = ReverseEcholocationBootstrapper(frames, performance_keyframes)
    bootstrapper.aqntl = initial_tempo  # Use the caller's initial tempo estimate
    return bootstrapper.run()