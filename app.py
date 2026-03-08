import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify
import librosa

app = Flask(__name__)

# ─── Health endpoint — keeps Render free tier awake via UptimeRobot ping ───
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

# ─── Main analysis endpoint ────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    suffix = os.path.splitext(file.filename)[1] or '.mp3'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        # Load mono, cap at 3 min to keep Render free tier happy on memory
        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=180)

        # ── BPM ──────────────────────────────────────────────────────────
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(round(float(tempo), 1))

        # ── Key detection ─────────────────────────────────────────────────
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key_index = int(np.argmax(chroma_mean))

        major_template = np.array([1,0,1,0,1,1,0,1,0,1,0,1], dtype=float)
        minor_template = np.array([1,0,1,1,0,1,0,1,1,0,1,0], dtype=float)
        major_score = float(np.dot(chroma_mean, np.roll(major_template, key_index)))
        minor_score = float(np.dot(chroma_mean, np.roll(minor_template, key_index)))
        mode = 'major' if major_score >= minor_score else 'minor'
        key_full = f"{key_names[key_index]} {mode}"

        # ── Loudness ──────────────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_db = float(round(20 * np.log10(rms_mean + 1e-9), 1))
        peak = float(np.max(np.abs(y)))
        peak_db = float(round(20 * np.log10(peak + 1e-9), 1))
        crest_factor = float(round(peak_db - rms_db, 1))

        # ── Spectral features ─────────────────────────────────────────────
        centroid  = float(round(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]), 1))
        rolloff   = float(round(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]), 1))
        bandwidth = float(round(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]), 1))
        zcr       = float(round(np.mean(librosa.feature.zero_crossing_rate(y)[0]), 4))

        # ── Frequency band energy % ───────────────────────────────────────
        stft  = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        def band_energy(low, high):
            mask = (freqs >= low) & (freqs < high)
            return float(np.mean(stft[mask, :])) if np.any(mask) else 0.0

        sub  = band_energy(20,   80)
        low  = band_energy(80,   300)
        mid  = band_energy(300,  2000)
        hi   = band_energy(2000, 8000)
        air  = band_energy(8000, sr / 2)
        tot  = sub + low + mid + hi + air + 1e-9

        bands = {
            'sub_bass_20_80hz': round(sub / tot * 100, 1),
            'low_mid_80_300hz': round(low / tot * 100, 1),
            'mids_300hz_2khz':  round(mid / tot * 100, 1),
            'high_2_8khz':      round(hi  / tot * 100, 1),
            'air_8khz_plus':    round(air / tot * 100, 1),
        }

        # ── Onset strength (attack density) ──────────────────────────────
        onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = float(round(float(np.mean(onset_env)), 3))

        return jsonify({
            'bpm':                   bpm,
            'key':                   key_full,
            'rms_db':                rms_db,
            'peak_db':               peak_db,
            'crest_factor_db':       crest_factor,
            'spectral_centroid_hz':  centroid,
            'spectral_rolloff_hz':   rolloff,
            'spectral_bandwidth_hz': bandwidth,
            'zero_crossing_rate':    zcr,
            'onset_strength_mean':   onset_mean,
            'band_energy_percent':   bands,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
