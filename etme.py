"""
Electro-Thermodynamic Music Engine (ETME)
Master Loop: Boots up the physics simulation and processes raw MIDI data.
"""
from particle import Particle
from harmonic_canvas import HarmonicCanvas
from information_density import InformationDensityScanner


def run_etme_simulation(raw_midi_data):
    print("Initializing Electro-Thermodynamic Music Engine...\n")
    
    # Load particles
    particles = [Particle(pitch, vel, onset, dur) for (pitch, vel, onset, dur) in raw_midi_data]
    
    # PASS 1: Paint the Harmonic Canvas
    canvas = HarmonicCanvas(window_size_ms=400)
    regimes = canvas.process_timeline(particles)
    
    print("--- PASS 1: HARMONIC REGIMES ESTABLISHED ---")
    for r in regimes[:3]:  # Print first 3
        print(f"  Time {r['start_time']}ms -> State: {r['state']} | Frequencies: {r['active_pitches']}")

    # PASS 2: Scan for Information Density (Melody Extraction)
    scanner = InformationDensityScanner()
    scored_particles = scanner.calculate_id_scores(particles)
    
    print("\n--- PASS 2: INFORMATION DENSITY FILTER APPLIED ---")
    melodies = [p for p in scored_particles if "Voice 1" in p.voice_tag]
    background = [p for p in scored_particles if "Background" in p.voice_tag]
    
    print(f"  Detected {len(melodies)} high-entropy Melody particles.")
    print(f"  Detected {len(background)} low-entropy Harmonic Background particles.")
    
    # Print detailed particle breakdown
    print("\n--- PARTICLE DETAIL ---")
    for p in scored_particles:
        print(f"  {p}")

    return {
        "regimes": regimes,
        "melodies": melodies,
        "background": background,
        "all_particles": scored_particles
    }


# ---------------------------------------------------------
# Test with mock data: slow repeating bass (low entropy) 
# vs. fast right-hand run (high entropy)
# ---------------------------------------------------------
if __name__ == "__main__":
    mock_data = [
        # Bass (C2, quiet, slow, repeating)
        (36, 40, 0, 500), (36, 40, 500, 500), (36, 40, 1000, 500),
        # Melody (C5 -> D5 -> E5, loud, fast, moving)
        (72, 90, 0, 150), (74, 90, 150, 150), (76, 90, 300, 150)
    ]

    run_etme_simulation(mock_data)
