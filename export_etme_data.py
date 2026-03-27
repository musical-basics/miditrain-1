"""
Exports ETME Phase 1 + Phase 2 analysis of a real MIDI file as JSON
for the browser-based piano roll visualizer.

Phase 1 uses the HarmonicRegimeDetector from STS_bootstrapper.py 
(vector-based color wheel with HSL output).
"""
import json
from symusic import Score
from particle import Particle
from STS_bootstrapper import HarmonicRegimeDetector, SEMITONE_MAP
from information_density import InformationDensityScanner

# Map MIDI pitch class (0-11) to interval names for the regime detector
PC_TO_INTERVAL = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"
}


def midi_to_particles(midi_path):
    """Convert a real MIDI file into Particles."""
    score = Score(midi_path)
    tpq = score.ticks_per_quarter
    tick_to_ms = 500.0 / tpq

    particles = []
    for track in score.tracks:
        for note in track.notes:
            particles.append(Particle(
                pitch=note.pitch,
                velocity=note.velocity,
                onset_ms=int(note.start * tick_to_ms),
                duration_ms=int(note.duration * tick_to_ms)
            ))

    particles.sort(key=lambda p: p.onset)
    return particles


def extract_keyframes(midi_path):
    """Convert MIDI into keyframes for the HarmonicRegimeDetector.
    Returns list of (time_ms, [(interval_name, octave, velocity), ...])
    """
    score = Score(midi_path)
    tpq = score.ticks_per_quarter
    tick_to_ms = 500.0 / tpq

    time_map = {}
    for track in score.tracks:
        for note in track.notes:
            time_ms = int(note.start * tick_to_ms)
            interval = PC_TO_INTERVAL[note.pitch % 12]
            octave = note.pitch // 12
            if time_ms not in time_map:
                time_map[time_ms] = []
            time_map[time_ms].append((interval, octave, note.velocity))

    return [(t, time_map[t]) for t in sorted(time_map.keys())]


def export_analysis(midi_path, output_json="etme_analysis.json"):
    print(f"Loading MIDI: {midi_path}")
    particles = midi_to_particles(midi_path)
    keyframes = extract_keyframes(midi_path)
    print(f"  Loaded {len(particles)} particles, {len(keyframes)} keyframes")

    # =============================================
    # Phase 1: HarmonicRegimeDetector (from STS_bootstrapper)
    # =============================================
    print("Running Phase 1: Harmonic Regime Detector...")
    detector = HarmonicRegimeDetector(buffer_ms=300, debounce_ms=150)

    regime_frames = []
    for time_ms, notes in keyframes:
        frame = detector.process_frame(time_ms, notes)
        regime_frames.append(frame)

    # Build contiguous regime blocks from consecutive same-state frames
    regimes = []
    current_regime = None
    for frame in regime_frames:
        state = frame["State"]
        if current_regime is None or current_regime["state"] != state:
            if current_regime:
                current_regime["end_time"] = frame["Time (ms)"]
                regimes.append(current_regime)
            current_regime = {
                "start_time": frame["Time (ms)"],
                "end_time": frame["Time (ms)"],
                "state": state,
                "hue": frame["Hue"],
                "saturation": frame["Sat (%)"],
                "v_vec": frame["V_vec"]
            }
        else:
            # Update with latest values within the same regime
            current_regime["end_time"] = frame["Time (ms)"]
            current_regime["hue"] = frame["Hue"]
            current_regime["saturation"] = frame["Sat (%)"]
    if current_regime:
        # Extend last regime to cover the last note
        current_regime["end_time"] = particles[-1].onset + particles[-1].duration
        regimes.append(current_regime)

    print(f"  Detected {len(regimes)} harmonic regimes")
    state_counts = {}
    for r in regimes:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
    for s, c in state_counts.items():
        print(f"    {s}: {c}")

    # Store per-frame data so we can color each note by its exact hue
    frame_lookup = []
    for frame in regime_frames:
        frame_lookup.append({
            "time": frame["Time (ms)"],
            "hue": frame["Hue"],
            "sat": frame["Sat (%)"],
            "v_vec": frame["V_vec"],
            "state": frame["State"]
        })

    # =============================================
    # Phase 2: Information Density
    # =============================================
    print("Running Phase 2: Information Density...")
    scanner = InformationDensityScanner(melody_threshold=50.0)
    scored_particles = scanner.calculate_id_scores(particles)

    melodies = [p for p in scored_particles if "Voice 1" in p.voice_tag]
    print(f"  Tagged {len(melodies)} melody particles")

    # Build JSON output
    notes_json = []
    for p in scored_particles:
        # Find the closest regime frame for this note's onset
        closest_frame = min(frame_lookup, key=lambda f: abs(f["time"] - p.onset))
        notes_json.append({
            "pitch": p.pitch,
            "velocity": p.velocity,
            "onset": p.onset,
            "duration": p.duration,
            "id_score": round(p.id_score, 2),
            "voice_tag": p.voice_tag,
            # Phase 1 color data per note
            "hue": closest_frame["hue"],
            "sat": closest_frame["sat"],
            "regime_state": closest_frame["state"]
        })

    regimes_json = []
    for r in regimes:
        regimes_json.append({
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "state": r["state"],
            "hue": r["hue"],
            "saturation": r["saturation"],
            "v_vec": r["v_vec"]
        })

    data = {
        "notes": notes_json,
        "regimes": regimes_json,
        "stats": {
            "total_notes": len(notes_json),
            "total_regimes": len(regimes_json),
            "melody_notes": len(melodies),
            "background_notes": len(notes_json) - len(melodies)
        }
    }

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Analysis exported to: {output_json}")
    return data


if __name__ == "__main__":
    export_analysis("raw_midi_grieg_waltz.mid")
