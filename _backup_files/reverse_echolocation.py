class Anchor:
    def __init__(self, measure, beat, time_ms, state_note=""):
        self.measure = measure
        self.beat = beat
        self.time_ms = time_ms
        self.state_note = state_note 

    def __repr__(self):
        return f"M{self.measure} B{self.beat} @ {self.time_ms:.1f}ms - [{self.state_note}]"


class FixedReverseEcholocation:
    def __init__(self, regime_frames, initial_aqntl_ms=500.0, beats_per_measure=4):
        self.frames = regime_frames 
        self.aqntl = initial_aqntl_ms 
        self.anchors = []
        
        self.last_anchor_time = 0.0
        self.current_measure = 1
        self.current_beat = 1
        self.beats_per_measure = beats_per_measure
        
        # We index frames by time for safer lookups (fixes the infinite loop bug)
        self.max_time = float(self.frames[-1]["Time (ms)"]) if self.frames else 0

    def run(self):
        if not self.frames:
            return self.anchors

        # 1. BOOTSTRAP: Find the first chord strike to anchor Measure 1, Beat 1
        seed_time = self._find_next_spike(0)
        if seed_time is None:
            print("No harmonic transitions found to bootstrap.")
            return []

        self._lock_anchor(seed_time, "Seed Anchor", update_tempo=False)

        # 2. THE METRICAL LOOP: Walk forward through time using our tempo (AQNTL)
        while True:
            expected_time = self.last_anchor_time + self.aqntl
            
            if expected_time > self.max_time:
                break # Reached the end of the performance

            # Calculate safe "On-Beat" window (+/- 20% of current tempo)
            # Example: At 500ms tempo, this is a 100ms window in either direction
            standard_buffer = self.aqntl * 0.20 
            
            # Look for a harmonic transition within the safe window
            spike_time = self._find_spike_in_window(
                expected_time - standard_buffer, 
                expected_time + standard_buffer
            )

            # SCENARIO A: Chord Change on the Beat (Rubato Adjuster)
            if spike_time is not None:
                # We snap the metrical grid perfectly to the chord change
                # AND we update the AQNTL tempo tracker so it speeds up/slows down with the pianist
                self._lock_anchor(spike_time, "On-Beat Transition", update_tempo=True)
                continue

            # SCENARIO B: No Chord Change. Check the current harmonic state.
            state_at_expected = self._get_state_at_time(expected_time)

            if state_at_expected == "Regime Locked":
                # The pianist is just holding a chord over the beat! 
                # Confidently dead-reckon the beat. DO NOT update tempo.
                self._lock_anchor(expected_time, "Held Chord (Dead-Reckon)", update_tempo=False)
                continue

            elif state_at_expected in ["Undefined / Gray Void", "Silence"]:
                # SCENARIO C: The Fermata / Rest / Chaos State
                # The performer paused or is playing an unmetered cadenza.
                print(f"Void detected at M{self.current_measure} B{self.current_beat}. Initiating Fermata Resync...")
                
                next_spike_time = self._find_next_spike(expected_time)
                
                if next_spike_time is None:
                    break # Piece ended on a fade out
                
                # --- THE FERMATA MATH ---
                # Calculate how much time passed in the void, and divide by our last known tempo 
                # to figure out exactly how many musical beats we skipped.
                time_in_void = next_spike_time - self.last_anchor_time
                beats_skipped = round(time_in_void / self.aqntl)
                
                if beats_skipped > 1:
                    self._advance_grid_by_beats(beats_skipped - 1)
                    
                self._lock_anchor(next_spike_time, "Fermata Resync", update_tempo=False)
                continue

        return self.anchors

    # --- HELPER METHODS ---

    def _lock_anchor(self, time_ms, note, update_tempo=True):
        """Locks the beat and handles Rubato tracking via EMA."""
        self.anchors.append(Anchor(self.current_measure, self.current_beat, time_ms, state_note=note))
        
        if update_tempo:
            instant_tempo = time_ms - self.last_anchor_time
            # EMA: 70% momentum, 30% new data (smooths out human error)
            self.aqntl = (self.aqntl * 0.7) + (instant_tempo * 0.3) 
            
        self.last_anchor_time = time_ms
        self._advance_grid_by_beats(1)

    def _advance_grid_by_beats(self, num_beats):
        """Mathematically advances the XML grid, rolling over measures perfectly."""
        total_beats = (self.current_beat - 1) + num_beats
        
        measures_added = total_beats // self.beats_per_measure
        new_beat = (total_beats % self.beats_per_measure) + 1
        
        self.current_measure += measures_added
        self.current_beat = new_beat

    def _get_state_at_time(self, target_time):
        """Finds the closest frame to the expected time."""
        closest_frame = min(self.frames, key=lambda f: abs(float(f["Time (ms)"]) - target_time))
        return closest_frame["State"]

    def _find_spike_in_window(self, start_ms, end_ms):
        """Searches strictly within a time bounds for a Transition Spike."""
        for frame in self.frames:
            t = float(frame["Time (ms)"])
            if start_ms <= t <= end_ms and frame["State"] == "TRANSITION SPIKE!":
                return t
        return None

    def _find_next_spike(self, start_ms):
        """Unbounded forward scan for Fermatas/Bootstrapping."""
        for frame in self.frames:
            t = float(frame["Time (ms)"])
            if t > start_ms and frame["State"] == "TRANSITION SPIKE!":
                return t
        return None

# ==========================================
# SIMULATION: Rubato & Held Chords
# ==========================================
if __name__ == "__main__":
    # Simulated output mimicking a pianist playing at ~500ms tempo
    mock_frames = [
        {"Time (ms)": 0, "State": "Silence"},
        {"Time (ms)": 100, "State": "TRANSITION SPIKE!"}, # M1 B1 (Seed)
        {"Time (ms)": 150, "State": "Regime Locked"},
        
        # M1 B2: No transition! The pianist just holds the chord through beat 2.
        {"Time (ms)": 600, "State": "Regime Locked"}, 
        
        # M1 B3: Pianist speeds up slightly (Rubato: hits at 1050ms instead of 1100ms)
        {"Time (ms)": 1050, "State": "TRANSITION SPIKE!"}, 
        {"Time (ms)": 1100, "State": "Regime Locked"},
        
        # M1 B4: Syncopated chord anticipation! Played way too early (1300ms)
        # The algorithm should IGNORE this spike for metrical mapping and dead-reckon M1 B4 instead
        {"Time (ms)": 1300, "State": "TRANSITION SPIKE!"},
        {"Time (ms)": 1500, "State": "Regime Locked"},
        
        # The Fermata (Total Silence for over 2 seconds)
        {"Time (ms)": 1800, "State": "Silence"},
        {"Time (ms)": 3000, "State": "Silence"},
        
        # M2 B4: The pianist crashes back in on beat 4 after the fermata.
        {"Time (ms)": 3480, "State": "TRANSITION SPIKE!"},
    ]

    bootstrapper = FixedReverseEcholocation(mock_frames, initial_aqntl_ms=500.0, beats_per_measure=4)
    generated_grid = bootstrapper.run()
    
    print("\n--- Implied XML Grid Generated ---")
    for anchor in generated_grid:
        print(anchor)