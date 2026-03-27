"""
Exports ETME Phase 1 + Phase 2 analysis of a real MIDI file as JSON
for the browser-based piano roll visualizer.
"""
import json
from symusic import Score
from particle import Particle
from harmonic_canvas import HarmonicCanvas
from information_density import InformationDensityScanner


def midi_to_particles(midi_path):
    """Convert a real MIDI file into Particles."""
    score = Score(midi_path)
    tpq = score.ticks_per_quarter
    tick_to_ms = 500.0 / tpq  # Assuming ~120 BPM

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


def export_analysis(midi_path, output_json="etme_analysis.json"):
    print(f"Loading MIDI: {midi_path}")
    particles = midi_to_particles(midi_path)
    print(f"  Loaded {len(particles)} particles")

    # Phase 1: Harmonic Canvas
    print("Running Phase 1: Harmonic Canvas...")
    canvas = HarmonicCanvas(window_size_ms=400)
    regimes = canvas.process_timeline(particles)
    print(f"  Detected {len(regimes)} harmonic regimes")

    # Phase 2: Information Density
    print("Running Phase 2: Information Density...")
    scanner = InformationDensityScanner(melody_threshold=50.0)
    scored_particles = scanner.calculate_id_scores(particles)

    melodies = [p for p in scored_particles if "Voice 1" in p.voice_tag]
    print(f"  Tagged {len(melodies)} melody particles")

    # Build JSON output
    notes_json = []
    for p in scored_particles:
        notes_json.append({
            "pitch": p.pitch,
            "velocity": p.velocity,
            "onset": p.onset,
            "duration": p.duration,
            "id_score": round(p.id_score, 2),
            "voice_tag": p.voice_tag
        })

    regimes_json = []
    for r in regimes:
        regimes_json.append({
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "pitches": r["active_pitches"],
            "state": r["state"]
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
