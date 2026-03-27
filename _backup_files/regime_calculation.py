import math
from collections import deque

# The 12 relative interval nodes (Eb is our hypothetical "Root" / Pure Red for this test)
INTERVAL_ANGLES = {
    "1": 0, "b2": 180, "2": 120, "b3": 270, "3": 60, "4": 330,
    "#4": 210, "5": 30, "b6": 300, "6": 90, "b7": 240, "7": 150
}

class HarmonicRegimeDetector:
    def __init__(self, buffer_ms=300):
        self.buffer_ms = buffer_ms
        # Deque stores tuples of: (time_ms, [list_of_active_notes])
        self.history = deque()
        self.prev_x = 0.0
        self.prev_y = 0.0

    def process_frame(self, current_time_ms, active_notes):
        """
        Processes a single frame in time, updating the 300ms buffer 
        and calculating the dynamic regime state.
        """
        # 1. Add current frame to history
        self.history.append((current_time_ms, active_notes))
        
        # 2. Purge old data outside the 300ms rolling window
        while self.history and self.history[0][0] < current_time_ms - self.buffer_ms:
            self.history.popleft()

        # 3. Calculate the weighted vector average of the ENTIRE 300ms buffer
        x_total, y_total, weight_total = 0.0, 0.0, 0.0
        
        for _, frame_notes in self.history:
            for interval, octave, velocity in frame_notes:
                if velocity <= 0: continue
                weight = velocity / 127.0
                weight_total += weight
                
                angle_rad = math.radians(INTERVAL_ANGLES[interval])
                x_total += weight * math.cos(angle_rad)
                y_total += weight * math.sin(angle_rad)

        # Failsafe for silence
        if weight_total == 0:
            self.prev_x, self.prev_y = 0.0, 0.0
            return {"Time": current_time_ms, "Hue": 0.0, "Sat": 0.0, "V_vec": 0.0, "State": "Silence"}

        x_avg = x_total / weight_total
        y_avg = y_total / weight_total

        # 4. Calculate Vector Velocity (Speed of change from last frame)
        v_vec = math.sqrt((x_avg - self.prev_x)**2 + (y_avg - self.prev_y)**2) * 100.0
        
        # Save current coordinates for the next frame's comparison
        self.prev_x, self.prev_y = x_avg, y_avg

        # 5. Calculate Final HSL
        final_hue = math.degrees(math.atan2(y_avg, x_avg))
        if final_hue < 0: final_hue += 360
        final_saturation = math.sqrt(x_avg**2 + y_avg**2) * 100.0

        # 6. Determine Regime State Logic
        state = "Stable"
        if final_saturation < 30.0:
            state = "Undefined / Gray Void"
        elif v_vec > 15.0:
            state = "TRANSITION SPIKE!"
        elif final_saturation > 75.0 and v_vec < 5.0:
            state = "Regime Locked"

        return {
            "Time (ms)": f"{current_time_ms:04d}",
            "Hue": round(final_hue, 1),
            "Sat (%)": round(final_saturation, 1),
            "V_vec (Speed)": round(v_vec, 1),
            "State": state
        }

# ==========================================
# SIMULATION: Beethoven's Pathétique, M. 2
# ==========================================
detector = HarmonicRegimeDetector(buffer_ms=300)

# We map a timeline of events based on the sheet music you provided.
# "1" = Eb (Root), "3" = G, "5" = Bb
timeline = [
    # Time 0 to 40ms: The fast, dissonant Grace Notes (F, D natural, F) 
    # Clashing with residual buffer noise.
    (0, [("2", 5, 80), ("7", 4, 80)]), 
    (20, [("2", 5, 80), ("7", 4, 80)]),
    
    # Time 60ms: The Lonely Downbeat (Just the Bb melody note)
    (60, [("5", 5, 90)]),
    (80, [("5", 5, 90)]),
    
    # Time 120ms: The Anchor Drops! (LH strikes the full Eb Major chord)
    (120, [("1", 2, 110), ("3", 3, 100), ("5", 3, 100), ("5", 5, 90)]),
    (140, [("1", 2, 110), ("3", 3, 100), ("5", 3, 100), ("5", 5, 90)]),
    (180, [("1", 2, 110), ("3", 3, 100), ("5", 3, 100), ("5", 5, 90)])
]

print(f"{'Time':<10} | {'Hue':<6} | {'Sat (%)':<8} | {'V_vec (Speed)':<14} | {'State'}")
print("-" * 65)

for time_ms, notes in timeline:
    res = detector.process_frame(time_ms, notes)
    print(f"{res['Time (ms)']:<10} | {res['Hue']:<6} | {res['Sat (%)']:<8} | {res['V_vec (Speed)']:<14} | {res['State']}")