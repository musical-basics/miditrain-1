import math
from collections import deque, Counter

# ==========================================
# 1. THE HSL VECTOR PHYSICS ENGINE
# ==========================================
INTERVAL_ANGLES = {
    "1": 0, "b2": 180, "2": 120, "b3": 270, "3": 60, "4": 330,
    "#4": 210, "5": 30, "b6": 300, "6": 90, "b7": 240, "7": 150
}

class HarmonicRegimeDetector:
    def __init__(self, buffer_ms=300):
        self.buffer_ms = buffer_ms
        self.history = deque()
        self.prev_x = 0.0
        self.prev_y = 0.0

    def process_frame(self, current_time_ms, active_notes):
        self.history.append((current_time_ms, active_notes))
        
        while self.history and self.history[0][0] < current_time_ms - self.buffer_ms:
            self.history.popleft()

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

        x_avg = x_total / weight_total
        y_avg = y_total / weight_total

        v_vec = math.sqrt((x_avg - self.prev_x)**2 + (y_avg - self.prev_y)**2) * 100.0
        self.prev_x, self.prev_y = x_avg, y_avg

        final_hue = math.degrees(math.atan2(y_avg, x_avg))
        if final_hue < 0: final_hue += 360
        final_saturation = math.sqrt(x_avg**2 + y_avg**2) * 100.0

        state = "Stable"
        if final_saturation < 30.0:
            state = "Undefined / Gray Void"
        elif v_vec > 15.0:
            state = "TRANSITION SPIKE!"
        elif final_saturation > 75.0 and v_vec < 5.0:
            state = "Regime Locked"

        return {
            "Time (ms)": current_time_ms,
            "Hue": round(final_hue, 1),
            "Sat (%)": round(final_saturation, 1),
            "V_vec": round(v_vec, 1),
            "State": state
        }


# ==========================================
# 1.5 SUBDIVISION SCANNER (EDGE CASE 1)
# ==========================================
class SubdivisionScanner:
    @staticmethod
    def get_grid_resolution(keyframes, window_ms=3000):
        # Extract onset times where actual notes are struck
        onsets = [k[0] for k in keyframes if k[0] <= window_ms and len(k[1]) > 0]
        if len(onsets) < 2: return None
        
        deltas = [onsets[i] - onsets[i-1] for i in range(1, len(onsets))]
        rounded_deltas = [round(d, -1) for d in deltas if d > 20]
        
        if not rounded_deltas: return None
        
        counter = Counter(rounded_deltas)
        most_common_gaps = [g for g, count in counter.most_common(3)]
        return min(most_common_gaps)


# ==========================================
# 2. THE METRICAL GRID BOOTSTRAPPER
# ==========================================
class Anchor:
    def __init__(self, measure, beat, time_ms, state_note=""):
        self.measure = measure
        self.beat = beat
        self.time_ms = time_ms
        self.state_note = state_note 

    def __repr__(self):
        return f"M{self.measure} B{self.beat} @ {self.time_ms:04d}ms | {self.state_note}"

class ReverseEcholocationBootstrapper:
    def __init__(self, regime_frames, initial_aqntl_ms=500.0, beats_per_measure=4):
        self.frames = regime_frames 
        self.aqntl = initial_aqntl_ms 
        self.anchors = []
        self.last_anchor_time = 0.0
        self.current_measure = 1
        self.current_beat = 1
        self.beats_per_measure = beats_per_measure
        self.max_time = float(self.frames[-1]["Time (ms)"]) if self.frames else 0

    def run(self, keyframes=None):
        if not self.frames: return []

        # ---------------------------------------------------------
        # 1. BOOTSTRAP: Anacrusis (Pickup) Detector
        # ---------------------------------------------------------
        first_spike = self._find_next_spike(0)
        if first_spike is None: return []

        if keyframes:
            grid_res = SubdivisionScanner.get_grid_resolution(keyframes)
            if grid_res: print(f"Detected underlying rhythmic subdivision: ~{grid_res}ms")

            second_spike = self._find_next_spike(first_spike + 100)
            
            # Helper to calculate the "heaviness" of a chord strike
            def get_weight(target_time):
                return sum(sum(n[2] for n in notes) for t, notes in keyframes if abs(t - target_time) < 50)
                
            first_weight = get_weight(first_spike)
            second_weight = get_weight(second_spike) if second_spike else 0
            
            if second_spike and second_weight > (first_weight * 1.5):
                print("-> Anacrusis (Pickup) Detected! First spike is lighter than the second.")
                distance_ms = second_spike - first_spike
                beats_backward = round(distance_ms / self.aqntl)
                pickup_beat = self.beats_per_measure - beats_backward + 1
                
                self.current_measure, self.current_beat = 0, int(pickup_beat)
                self._lock_anchor(first_spike, "Pickup Anchor", update_tempo=False)
                
                self.current_measure, self.current_beat = 1, 1
                self._lock_anchor(second_spike, "True Downbeat", update_tempo=False)
            else:
                self._lock_anchor(first_spike, "Seed Anchor (M1 B1)", update_tempo=False)
        else:
            self._lock_anchor(first_spike, "Seed Anchor (M1 B1)", update_tempo=False)


        # ---------------------------------------------------------
        # 2. METRICAL LOOP
        # ---------------------------------------------------------
        while True:
            expected_time = self.last_anchor_time + self.aqntl
            if expected_time > self.max_time: break 

            standard_buffer = self.aqntl * 0.20 
            syncopation_buffer = self.aqntl * 0.50 # Edge Case 2 boundary
            
            spike_time = self._find_spike_in_window(
                expected_time - standard_buffer, 
                expected_time + standard_buffer
            )

            # A. Chord Change on the Beat (Rubato tracked)
            if spike_time is not None:
                self._lock_anchor(spike_time, "On-Beat Transition", update_tempo=True)
                continue
                
            # B. Edge Case 2: SYNCOPATION TRAP
            early_spike = self._find_spike_in_window(
                expected_time - syncopation_buffer, 
                expected_time - standard_buffer
            )
            
            if early_spike is not None:
                self._lock_anchor(early_spike, "Syncopated Anticipation", update_tempo=False)
                self.last_anchor_time = expected_time # Push grid forward mathematically
                continue

            state_at_expected = self._get_state_at_time(expected_time)

            # C. Held Chord
            if state_at_expected == "Regime Locked":
                self._lock_anchor(expected_time, "Held Chord (Dead-Reckon)", update_tempo=False)
                continue

            # D. Edge Case 3: Fermata / Void Math
            if state_at_expected in ["Undefined / Gray Void", "Silence"]:
                print(f"-> Void detected at M{self.current_measure} B{self.current_beat}. Initiating Fermata Resync...")
                
                post_fermata_spike = self._find_next_spike(expected_time)
                if post_fermata_spike is None: break 
                
                # Look-Ahead for Waking Tempo
                second_post_fermata_spike = self._find_next_spike(post_fermata_spike + 50)
                if second_post_fermata_spike is not None:
                    new_waking_tempo = second_post_fermata_spike - post_fermata_spike
                    effective_tempo = (self.aqntl + new_waking_tempo) / 2.0
                else:
                    new_waking_tempo = None
                    effective_tempo = self.aqntl
                
                # Math & Snapping Heuristic
                time_in_void = post_fermata_spike - self.last_anchor_time
                raw_beats_skipped = time_in_void / effective_tempo
                
                if raw_beats_skipped > self.beats_per_measure:
                    print("-> Massive pause detected. Snapping re-entry to the next Downbeat.")
                    beats_to_next_downbeat = (self.beats_per_measure - self.current_beat) + 1
                    measures_skipped = int(raw_beats_skipped // self.beats_per_measure)
                    beats_skipped = beats_to_next_downbeat + (measures_skipped * self.beats_per_measure)
                else:
                    beats_skipped = round(raw_beats_skipped)
                
                if beats_skipped > 1:
                    self._advance_grid_by_beats(beats_skipped - 1)
                    
                self._lock_anchor(post_fermata_spike, "Fermata Resync", update_tempo=False)
                
                # Instantly calibrate tempo for the next measure!
                if new_waking_tempo is not None:
                    self.aqntl = new_waking_tempo
                    
                continue

        return self.anchors

    def _lock_anchor(self, time_ms, note, update_tempo=True):
        self.anchors.append(Anchor(self.current_measure, self.current_beat, int(time_ms), state_note=note))
        if update_tempo:
            instant_tempo = time_ms - self.last_anchor_time
            self.aqntl = (self.aqntl * 0.7) + (instant_tempo * 0.3) 
        self.last_anchor_time = time_ms
        self._advance_grid_by_beats(1)

    def _advance_grid_by_beats(self, num_beats):
        total_beats = (self.current_beat - 1) + num_beats
        self.current_measure += int(total_beats // self.beats_per_measure)
        self.current_beat = int((total_beats % self.beats_per_measure) + 1)

    def _get_state_at_time(self, target_time):
        closest_frame = min(self.frames, key=lambda f: abs(f["Time (ms)"] - target_time))
        return closest_frame["State"]

    def _find_spike_in_window(self, start_ms, end_ms):
        for frame in self.frames:
            if start_ms <= frame["Time (ms)"] <= end_ms and frame["State"] == "TRANSITION SPIKE!":
                return frame["Time (ms)"]
        return None

    def _find_next_spike(self, start_ms):
        for frame in self.frames:
            if frame["Time (ms)"] > start_ms and frame["State"] == "TRANSITION SPIKE!":
                return frame["Time (ms)"]
        return None


# ==========================================
# 3. COMBINED PIPELINE RUNNER
# ==========================================
def run_full_pipeline(performance_keyframes, initial_tempo=500.0):
    detector = HarmonicRegimeDetector(buffer_ms=300)
    frames = []
    
    max_time = performance_keyframes[-1][0] + 500 
    current_idx = 0
    active_notes = []

    for time_ms in range(0, max_time + 10, 10):
        while current_idx < len(performance_keyframes) and performance_keyframes[current_idx][0] <= time_ms:
            active_notes = performance_keyframes[current_idx][1]
            current_idx += 1
            
        frame_data = detector.process_frame(time_ms, active_notes)
        frames.append(frame_data)

    bootstrapper = ReverseEcholocationBootstrapper(frames, initial_aqntl_ms=initial_tempo, beats_per_measure=4)
    # Pass keyframes to enable the Anacrusis scanner
    grid = bootstrapper.run(keyframes=performance_keyframes) 
    
    return grid


# ==========================================
# EXECUTION: The Ultimate Rubato/Fermata Test
# ==========================================
if __name__ == "__main__":
    live_performance = [
        (0, []),
        (100, [("1", 4, 100), ("3", 4, 100), ("5", 4, 100)]),
        (1020, [("4", 4, 100), ("6", 4, 100), ("1", 5, 100)]),
        (1250, [("b2", 4, 127), ("#4", 4, 127), ("7", 4, 127)]),
        (1600, []),
        (3550, [("1", 4, 120), ("5", 4, 120)])
    ]

    print("--- RAW PERFORMANCE DATA PROCESSED ---")
    print("Generating HSL Vectors... 10ms Frame Integration Complete.")
    print("Bootstrapping XML Grid...\n")
    
    final_xml_grid = run_full_pipeline(live_performance, initial_tempo=500.0)
    
    print("\n--- FINAL IMPLIED XML GRID ---")
    for anchor in final_xml_grid:
        print(anchor)