from particle import Particle


class HarmonicCanvas:
    """Phase 1: The Sustain Pedal + Valence Detector.

    Groups particles into time-windows and checks if their acoustic
    frequencies bond (Stateful) or repel (Stateless).
    """
    def __init__(self, window_size_ms=400):
        self.window_size = window_size_ms
        self.regimes = []  # Stores our continuous color blocks

    def _calculate_valence(self, cluster_pitches):
        """
        Checks the interval symmetry to determine if the cluster is 
        a Solid/Liquid (Stateful) or a Gas (Stateless/Portal).
        """
        # Simplified heuristic: perfectly symmetrical chords (diminished 7ths, 
        # augmented) have specific interval patterns. 
        # In a full build, this would map the exact acoustic beat frequencies.
        intervals = [(cluster_pitches[i] - cluster_pitches[i-1]) % 12 
                      for i in range(1, len(cluster_pitches))]
        
        # If all intervals are 3 (dim) or 4 (aug), it's stateless gas.
        if intervals and (all(i == 3 for i in intervals) or all(i == 4 for i in intervals)):
            return "Stateless (Phase-Change Portal)"
        return "Stateful (Solid/Liquid Ground)"

    def process_timeline(self, particles):
        """
        Sweeps the timeline, grouping notes into Harmonic Regimes.
        """
        # Sort particles chronologically
        particles.sort(key=lambda p: p.onset)
        
        current_window_start = 0
        current_cluster = []
        
        for p in particles:
            if p.onset < current_window_start + self.window_size:
                current_cluster.append(p)
            else:
                # Window closed. Analyze the mass we collected.
                if current_cluster:
                    pitches = sorted(list(set([n.pitch for n in current_cluster])))
                    valence = self._calculate_valence(pitches)
                    
                    self.regimes.append({
                        "start_time": current_window_start,
                        "end_time": p.onset,
                        "active_pitches": pitches,
                        "state": valence
                    })
                
                # Start new window
                current_window_start = p.onset
                current_cluster = [p]
                
        # Don't forget the last cluster
        if current_cluster:
            pitches = sorted(list(set([n.pitch for n in current_cluster])))
            valence = self._calculate_valence(pitches)
            self.regimes.append({
                "start_time": current_window_start,
                "end_time": current_cluster[-1].onset + current_cluster[-1].duration,
                "active_pitches": pitches,
                "state": valence
            })

        return self.regimes
