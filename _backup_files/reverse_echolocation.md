class Anchor:
    def __init__(self, measure, beat, time_ms, is_ghost=False, state_note=""):
        self.measure = measure
        self.beat = beat
        self.time_ms = time_ms
        self.is_ghost = is_ghost
        self.state_note = state_note # Just for logging (e.g., "Dead-Reckoned", "Transition")

    def __repr__(self):
        ghost_str = " [GHOST/PAUSE]" if self.is_ghost else ""
        return f"M{self.measure} B{self.beat} @ {self.time_ms:.1f}ms - {self.state_note}{ghost_str}"


class ReverseEcholocationBootstrapper:
    def __init__(self, regime_frames, initial_aqntl_ms=500.0, beats_per_measure=4):
        # The output frames from HarmonicRegimeDetector
        self.frames = regime_frames 
        
        # V5 State Machine Variables
        self.aqntl = initial_aqntl_ms # Starts at 0.5s (120 BPM)
        self.anchors = []
        self.consecutive_misses = 0 # Counter for triggering fresh-scan mode
        self.recent_outcomes = [] # Sliding window for runaway detection
        
        # Grid Tracking
        self.last_anchor_time = 0.0
        self.current_measure = 1
        self.current_beat = 1
        self.beats_per_measure = beats_per_measure
        self.frame_cursor = 0

    def run(self):
        """Processes all regime frames to build the implied XML grid."""
        if not self.frames:
            return self.anchors

        # 1. Bootstrap: Find the first "TRANSITION SPIKE!" to seed the first anchor
        seed_found = False
        while self.frame_cursor < len(self.frames):
            frame = self.frames[self.frame_cursor]
            if frame["State"] == "TRANSITION SPIKE!":
                self.last_anchor_time = float(frame["Time (ms)"])
                self.anchors.append(Anchor(self.current_measure, self.current_beat, self.last_anchor_time, state_note="Seed Anchor"))
                self._advance_grid()
                self.frame_cursor += 1
                seed_found = True
                break
            self.frame_cursor += 1

        if not seed_found:
            print("No harmonic transitions found to bootstrap the grid.")
            return []

        # 2. Step through the performance beat-by-beat (stepV5 logic)
        while self.frame_cursor < len(self.frames):
            
            # Check Runaway State (7 out of 10 bad outcomes)
            if self._is_runaway():
                print(f"Runaway detected at M{self.current_measure} B{self.current_beat}. Pausing for human intervention.")
                break
                
            expected_delta = self.aqntl
            expected_time = self.last_anchor_time + expected_delta
            
            # A. Calculate Scan Windows
            standard_buffer = expected_delta * 0.20 # +/- 20%
            wide_buffer = expected_delta * 0.50     # +/- 50%
            
            # B. Special Mode: Consecutive Misses / Fresh Scan
            if self.consecutive_misses >= 3:
                found_time = self._scan_forward_unbounded()
                if found_time is not None:
                    self._lock_anchor(found_time, "Fresh Scan Sync", update_aqntl=False) # Preserve AQNTL
                    self.consecutive_misses = 0
                    self._push_outcome("match")
                else:
                    break # End of piece
                continue

            # C. Standard Window Scan
            match_time = self._scan_window(expected_time - standard_buffer, expected_time + standard_buffer)
            
            if match_time is not None:
                # Good match found!
                self._lock_anchor(match_time, "Standard Match", update_aqntl=True)
                self.consecutive_misses = 0
                self._push_outcome("match")
                continue
                
            # D. Wide Window Scan
            wide_match_time = self._scan_window(expected_time - wide_buffer, expected_time + wide_buffer)
            
            if wide_match_time is not None:
                self._lock_anchor(wide_match_time, "Wide Match", update_aqntl=True)
                self.consecutive_misses = 0
                self._push_outcome("match")
                continue
                
            # E. Dead-Reckon Mode (Extrapolate from AQNTL)
            self.consecutive_misses += 1
            self._push_outcome("miss")
            
            # If we miss but the next beat is <= 2 beats away, we auto-advance
            if self.consecutive_misses <= 2:
                self._lock_anchor(expected_time, "Dead-Reckoned", update_aqntl=False)
            else:
                # Gap is > 2 beats, create a Ghost Anchor and pause
                ghost = Anchor(self.current_measure, self.current_beat, expected_time, is_ghost=True, state_note="Ghost Anchor")
                self.anchors.append(ghost)
                print(f"Lost harmonic tracking. Ghost anchor placed at {expected_time}ms.")
                # In the real app, this would pause the state machine here.
                # For this auto-script, we will auto-confirm it and keep going.
                self.last_anchor_time = expected_time
                self._advance_grid()

        return self.anchors

    # --- Helper Methods ---

    def _lock_anchor(self, time_ms, note, update_aqntl=True):
        """Locks the anchor and updates the exponential moving average for tempo."""
        self.anchors.append(Anchor(self.current_measure, self.current_beat, time_ms, state_note=note))
        
        if update_aqntl:
            instant_aqntl = time_ms - self.last_anchor_time
            # EMA: new = 0.7 * old + 0.3 * instant
            self.aqntl = (self.aqntl * 0.7) + (instant_aqntl * 0.3) 
            
        self.last_anchor_time = time_ms
        self._advance_grid()

    def _advance_grid(self):
        """Moves the internal musical grid forward by one beat."""
        self.current_beat += 1
        if self.current_beat > self.beats_per_measure:
            self.current_beat = 1
            self.current_measure += 1

    def _scan_window(self, start_ms, end_ms):
        """Searches the frames within a time window for a Transition Spike."""
        temp_cursor = self.frame_cursor
        while temp_cursor < len(self.frames):
            frame = self.frames[temp_cursor]
            t = float(frame["Time (ms)"])
            
            if t > end_ms:
                break # Past the window
                
            if start_ms <= t <= end_ms:
                if frame["State"] == "TRANSITION SPIKE!": #
                    self.frame_cursor = temp_cursor + 1
                    return t
            temp_cursor += 1
        return None

    def _scan_forward_unbounded(self):
        """Fresh pitch scan: Unbounded forward search for the next anchor."""
        while self.frame_cursor < len(self.frames):
            frame = self.frames[self.frame_cursor]
            if frame["State"] == "TRANSITION SPIKE!": #
                t = float(frame["Time (ms)"])
                self.frame_cursor += 1
                return t
            self.frame_cursor += 1
        return None

    def _push_outcome(self, outcome):
        """Append to sliding window, cap at 10 entries."""
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > 10:
            self.recent_outcomes.pop(0)

    def _is_runaway(self):
        """Returns true if >= 7 of the last 10 outcomes are misses."""
        if len(self.recent_outcomes) < 10:
            return False
        misses = sum(1 for o in self.recent_outcomes if o == "miss")
        return misses >= 7


# --- Simulation Example ---
if __name__ == "__main__":
    # Simulated output from your HarmonicRegimeDetector script
    mock_regime_frames = [
        {"Time (ms)": 0, "State": "Undefined / Gray Void"},
        {"Time (ms)": 120, "State": "TRANSITION SPIKE!"}, # Seed Anchor (M1 B1)
        {"Time (ms)": 140, "State": "Regime Locked"},
        # 500ms later... standard tempo
        {"Time (ms)": 625, "State": "TRANSITION SPIKE!"}, # M1 B2 Match
        {"Time (ms)": 650, "State": "Regime Locked"},
        # Pianist starts a ritardando... takes 700ms instead of 500ms
        {"Time (ms)": 1320, "State": "TRANSITION SPIKE!"}, # M1 B3 Wide Match!
        {"Time (ms)": 1340, "State": "Regime Locked"},
        # Total chaos/cadenza (Gray Void), system will Dead-Reckon M1 B4
        {"Time (ms)": 2000, "State": "Undefined / Gray Void"},
        {"Time (ms)": 2300, "State": "Undefined / Gray Void"},
        # Next major chord drops way late
        {"Time (ms)": 3200, "State": "TRANSITION SPIKE!"}, # Fresh Scan Sync M2 B1
    ]

    bootstrapper = ReverseEcholocationBootstrapper(mock_regime_frames, initial_aqntl_ms=500.0)
    generated_grid = bootstrapper.run()
    
    print("\n--- Implied XML Grid Generated ---")
    for anchor in generated_grid:
        print(anchor)