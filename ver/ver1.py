"""
Brainwave Audio Tone Generator
===============================
Production-level application for generating brainwave-entrainment audio.
Supports binaural beats, monaural beats, isochronic tones, and pure sine
waves with optional background noise, amplitude modulation, fade in/out,
soft limiting, and waveform preview.

Interfaces: CLI  (--cli)  and  GUI / Tkinter  (default)

Required dependencies:   numpy, scipy, matplotlib
Optional dependency:     soundfile   (FLAC export)

Examples
--------
  python brainwave_gen.py                                     # GUI
  python brainwave_gen.py --cli -m study --duration-min 10    # CLI
  python brainwave_gen.py --cli --generate-all                # batch
"""

# ═══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
import json
import logging
import argparse
import threading
import textwrap
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, List, Any, Callable

import numpy as np
from scipy.io import wavfile

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

# ═══════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BrainwaveGen")

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
DEF_SR = 44100
DEF_CARRIER = 200.0
DEF_VOLUME = 0.7
DEF_FADE = 0.5
DEF_OUTDIR = "generated_audio"
CFG_FILE = "session_config.json"

BRAINWAVE_MODES: Dict[str, Dict[str, Any]] = {
    "reading":       {"beat": 10.0, "band": "Alpha",   "label": "10 Hz Alpha – Reading"},
    "study":         {"beat": 14.0, "band": "Lo-Beta", "label": "14 Hz Low Beta – Study"},
    "deep_focus":    {"beat": 16.0, "band": "Beta",    "label": "16 Hz Beta – Deep Focus"},
    "gamma_focus":   {"beat": 40.0, "band": "Gamma",   "label": "40 Hz Gamma – Intense Focus"},
    "relax":         {"beat":  8.0, "band": "Alpha",   "label": "8 Hz Alpha – Relaxation"},
    "stress_relief": {"beat":  6.0, "band": "Theta",   "label": "6 Hz Theta – Stress Relief"},
    "sleep":         {"beat":  2.0, "band": "Delta",   "label": "2 Hz Delta – Sleep"},
    "meditation":    {"beat":  7.0, "band": "Theta",   "label": "7 Hz Theta – Meditation"},
    "creativity":    {"beat":  5.0, "band": "Theta",   "label": "5 Hz Theta – Creativity"},
    "memory_boost":  {"beat": 18.0, "band": "Beta",    "label": "18 Hz Beta – Memory Boost"},
}

TONE_TYPES = ("binaural", "monaural", "isochronic", "pure")
NOISE_TYPES = ("none", "white", "pink", "brown")


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION DATACLASS
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class AudioConfig:
    """Every parameter the generator exposes."""
    mode:            str   = "study"
    beat_freq:       float = 14.0
    carrier_freq:    float = DEF_CARRIER
    tone_type:       str   = "binaural"
    duration_min:    int   = 5
    duration_sec:    int   = 0
    volume:          float = DEF_VOLUME
    sample_rate:     int   = DEF_SR
    noise_type:      str   = "none"
    noise_intensity: float = 0.0
    fade_duration:   float = DEF_FADE
    output_filename: str   = ""
    output_dir:      str   = DEF_OUTDIR
    am_enabled:      bool  = False
    am_freq:         float = 1.0
    am_depth:        float = 0.5

    @property
    def total_seconds(self) -> float:
        return self.duration_min * 60.0 + self.duration_sec

    @property
    def total_samples(self) -> int:
        return int(self.total_seconds * self.sample_rate)

    @property
    def est_wav_bytes(self) -> int:
        """Stereo 16-bit PCM WAV = samples * 4 + 44-byte header."""
        return self.total_samples * 4 + 44

    def auto_filename(self) -> str:
        n = self.output_filename.strip()
        if not n:
            n = f"{self.mode}_{self.beat_freq:.0f}hz_{self.tone_type}.wav"
        if not n.lower().endswith((".wav", ".flac")):
            n += ".wav"
        return n

    def validate(self) -> List[str]:
        errors: List[str] = []
        dur = self.total_seconds
        nyq = self.sample_rate / 2.0
        if dur <= 0:
            errors.append("Duration must be > 0 seconds.")
        if dur > 3600:
            errors.append("Duration must be <= 60 minutes.")
        if not 0.0 <= self.volume <= 1.0:
            errors.append("Volume must be between 0.0 and 1.0.")
        if self.carrier_freq <= 0:
            errors.append("Carrier frequency must be > 0 Hz.")
        if self.carrier_freq >= nyq:
            errors.append(f"Carrier must be < {nyq:.0f} Hz (Nyquist).")
        if self.beat_freq < 0:
            errors.append("Beat frequency must be >= 0 Hz.")
        if self.beat_freq > 100:
            errors.append("Beat frequency should be <= 100 Hz.")
        if self.tone_type not in TONE_TYPES:
            errors.append(f"Tone type must be one of {TONE_TYPES}.")
        if self.noise_type not in NOISE_TYPES:
            errors.append(f"Noise type must be one of {NOISE_TYPES}.")
        if not 0.0 <= self.noise_intensity <= 1.0:
            errors.append("Noise intensity must be 0.0–1.0.")
        if self.fade_duration < 0:
            errors.append("Fade duration must be >= 0.")
        if dur > 0 and self.fade_duration > dur / 2:
            errors.append("Fade duration is too long for this audio duration.")
        if not 8000 <= self.sample_rate <= 192000:
            errors.append("Sample rate must be 8000–192000 Hz.")
        if self.tone_type == "binaural":
            lf = self.carrier_freq - self.beat_freq / 2.0
            rf = self.carrier_freq + self.beat_freq / 2.0
            if lf <= 0:
                errors.append(f"Left channel freq ({lf:.1f} Hz) must be > 0.")
            if rf >= nyq:
                errors.append(f"Right channel freq ({rf:.1f} Hz) >= Nyquist.")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ═══════════════════════════════════════════════════════════════════════
#  NOISE GENERATORS (DSP)
# ═══════════════════════════════════════════════════════════════════════
def _white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Flat power spectral density – each sample from N(0,1)."""
    return rng.standard_normal(n)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    1/f noise via spectral shaping.
    Equal energy per octave.  Shape white spectrum by 1/sqrt(f).
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1.0  # avoid division by zero at DC
    spectrum /= np.sqrt(freqs)  # power proportional to 1/f
    spectrum[0] = 0.0  # remove DC offset
    return np.fft.irfft(spectrum, n=n)


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    1/f^2 (Brownian) noise – integrated white noise.
    Linear drift removed for zero-mean result.
    """
    brown = np.cumsum(rng.standard_normal(n))
    brown -= np.linspace(brown[0], brown[-1], n)  # remove drift
    return brown


_NOISE_GENERATORS = {
    "white": _white_noise,
    "pink":  _pink_noise,
    "brown": _brown_noise,
}


def make_noise(kind: str, n: int) -> np.ndarray:
    """Return noise normalised to [-1, 1].  'none' returns zeros."""
    if kind == "none" or kind not in _NOISE_GENERATORS:
        return np.zeros(n)
    rng = np.random.default_rng()
    sig = _NOISE_GENERATORS[kind](n, rng)
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig


# ═══════════════════════════════════════════════════════════════════════
#  TONE GENERATORS (DSP)
# ═══════════════════════════════════════════════════════════════════════
def _time_axis(duration: float, sr: int) -> np.ndarray:
    """Create a time-sample array from 0 to duration at sample rate sr."""
    return np.arange(int(duration * sr), dtype=np.float64) / sr


def gen_sine(freq: float, duration: float, sr: int) -> np.ndarray:
    """Pure sine wave at the given frequency."""
    t = _time_axis(duration, sr)
    return np.sin(2.0 * np.pi * freq * t)


def gen_binaural(carrier: float, beat: float, duration: float, sr: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stereo binaural beat.
    Left  channel = carrier - beat/2
    Right channel = carrier + beat/2
    The brain perceives the frequency difference as the beat.
    """
    t = _time_axis(duration, sr)
    left_freq = carrier - beat / 2.0
    right_freq = carrier + beat / 2.0
    log.info("Binaural  L=%.2f Hz  R=%.2f Hz  delta=%.2f Hz",
             left_freq, right_freq, beat)
    left = np.sin(2.0 * np.pi * left_freq * t)
    right = np.sin(2.0 * np.pi * right_freq * t)
    return left, right


def gen_monaural(carrier: float, beat: float, duration: float, sr: int
                 ) -> np.ndarray:
    """
    Monaural beat – two close tones summed in one channel.
    The trigonometric identity produces AM at the beat frequency.
    sin(A) + sin(B) = 2 cos((A-B)/2) sin((A+B)/2)
    """
    t = _time_axis(duration, sr)
    f1 = carrier - beat / 2.0
    f2 = carrier + beat / 2.0
    log.info("Monaural  F1=%.2f Hz  F2=%.2f Hz  delta=%.2f Hz", f1, f2, beat)
    return 0.5 * (np.sin(2.0 * np.pi * f1 * t) +
                  np.sin(2.0 * np.pi * f2 * t))


def gen_isochronic(carrier: float, beat: float, duration: float, sr: int
                   ) -> np.ndarray:
    """
    Isochronic tone – carrier gated by a smooth pulse train at beat Hz.
    Envelope = max(0, cos(2*pi*beat*t)) gives smooth half-cosine pulses.
    """
    t = _time_axis(duration, sr)
    log.info("Isochronic  carrier=%.2f Hz  pulse=%.2f Hz", carrier, beat)
    tone = np.sin(2.0 * np.pi * carrier * t)
    if beat <= 0:
        return tone
    envelope = np.maximum(0.0, np.cos(2.0 * np.pi * beat * t))
    return tone * envelope


# ═══════════════════════════════════════════════════════════════════════
#  AUDIO PROCESSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════
def apply_fade(sig: np.ndarray, fade_secs: float, sr: int) -> np.ndarray:
    """
    Apply raised-cosine (half-Hann) fade-in and fade-out.
    Works on both mono (1-D) and stereo (2-D) arrays.
    """
    n = int(fade_secs * sr)
    if n <= 0 or 2 * n >= sig.shape[0]:
        return sig
    sig = sig.copy()
    # Raised cosine: 0 -> 1
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
    if sig.ndim == 2:
        sig[:n] *= ramp[:, None]
        sig[-n:] *= ramp[::-1, None]
    else:
        sig[:n] *= ramp
        sig[-n:] *= ramp[::-1]
    return sig


def apply_am(sig: np.ndarray, freq: float, depth: float, sr: int
             ) -> np.ndarray:
    """
    Amplitude modulation.
    out = sig * (1 - depth + depth * sin(2*pi*freq*t))
    depth=0 => no modulation, depth=1 => full modulation.
    """
    t = np.arange(sig.shape[0], dtype=np.float64) / sr
    modulator = 1.0 - depth + depth * np.sin(2.0 * np.pi * freq * t)
    if sig.ndim == 2:
        return sig * modulator[:, None]
    return sig * modulator


def soft_limit(sig: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Soft clipper using tanh above threshold.
    Prevents harsh digital clipping while preserving dynamics below threshold.
    """
    peak = np.max(np.abs(sig))
    if peak > threshold:
        sig = threshold * np.tanh(sig / threshold)
    return sig


def normalize(sig: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Peak-normalise the signal to the target amplitude."""
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig * (target / peak)
    return sig


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════
ProgressCB = Optional[Callable[[float, str], None]]


def generate_audio(cfg: AudioConfig, pcb: ProgressCB = None) -> np.ndarray:
    """
    Build a stereo float64 ndarray of shape (samples, 2) in [-1, 1].
    Applies tone generation, optional AM, noise mixing, volume,
    fade, soft limiting, and normalisation.
    """
    dur = cfg.total_seconds
    sr = cfg.sample_rate

    def _progress(value: float, message: str):
        if pcb:
            pcb(value, message)

    # ── Step 1: Generate the primary tone ───────────────────
    _progress(0.05, "Generating tone waveform…")

    if cfg.tone_type == "binaural":
        left_ch, right_ch = gen_binaural(
            cfg.carrier_freq, cfg.beat_freq, dur, sr)
        audio = np.column_stack((left_ch, right_ch))
    elif cfg.tone_type == "monaural":
        mono = gen_monaural(cfg.carrier_freq, cfg.beat_freq, dur, sr)
        audio = np.column_stack((mono, mono))
    elif cfg.tone_type == "isochronic":
        mono = gen_isochronic(cfg.carrier_freq, cfg.beat_freq, dur, sr)
        audio = np.column_stack((mono, mono))
    elif cfg.tone_type == "pure":
        mono = gen_sine(cfg.carrier_freq, dur, sr)
        audio = np.column_stack((mono, mono))
    else:
        raise ValueError(f"Unknown tone type: '{cfg.tone_type}'")

    # ── Step 2: Optional amplitude modulation ───────────────
    _progress(0.25, "Applying modulation…")
    if cfg.am_enabled and cfg.am_freq > 0:
        log.info("AM  freq=%.2f Hz  depth=%.2f", cfg.am_freq, cfg.am_depth)
        audio = apply_am(audio, cfg.am_freq, cfg.am_depth, sr)

    # ── Step 3: Mix in background noise ─────────────────────
    _progress(0.45, "Generating noise…")
    if cfg.noise_type != "none" and cfg.noise_intensity > 0:
        log.info("Noise: %s @ %.0f%%", cfg.noise_type,
                 cfg.noise_intensity * 100)
        n_samples = audio.shape[0]
        noise_left = make_noise(cfg.noise_type, n_samples)
        noise_right = make_noise(cfg.noise_type, n_samples)
        noise_stereo = np.column_stack((noise_left, noise_right))
        # Scale tone down slightly to keep it audible above noise
        tone_gain = 1.0 - 0.5 * cfg.noise_intensity
        audio = audio * tone_gain + noise_stereo * cfg.noise_intensity

    # ── Step 4: Volume, fade, limiter, normalise ────────────
    _progress(0.70, "Post-processing…")
    audio *= cfg.volume

    if cfg.fade_duration > 0:
        audio = apply_fade(audio, cfg.fade_duration, sr)

    audio = soft_limit(audio)
    audio = normalize(audio, target=min(0.95, max(cfg.volume, 0.1)))

    _progress(0.95, "Generation complete.")
    log.info("Generated %s samples x %d channels, peak=%.4f",
             f"{audio.shape[0]:,}", audio.shape[1], np.max(np.abs(audio)))
    return audio


# ═══════════════════════════════════════════════════════════════════════
#  FILE I/O
# ═══════════════════════════════════════════════════════════════════════
def save_wav(audio: np.ndarray, path: str, sr: int) -> str:
    """Save audio as 16-bit PCM stereo WAV file."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    # Convert float64 [-1,1] to int16
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, pcm)
    abs_path = os.path.abspath(path)
    log.info("WAV saved -> %s", abs_path)
    return abs_path


def save_flac(audio: np.ndarray, path: str, sr: int) -> Optional[str]:
    """Save audio as FLAC if soundfile is available."""
    if not HAS_SOUNDFILE:
        log.warning("soundfile not installed – FLAC export skipped.")
        return None
    flac_path = str(Path(path).with_suffix(".flac"))
    directory = os.path.dirname(os.path.abspath(flac_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    sf.write(flac_path, audio, sr, subtype="PCM_16")
    abs_path = os.path.abspath(flac_path)
    log.info("FLAC saved -> %s", abs_path)
    return abs_path


def save_config(cfg: AudioConfig, path: str = CFG_FILE):
    """Save current session configuration as JSON."""
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    log.info("Config saved -> %s", path)


def load_config(path: str = CFG_FILE) -> Optional[AudioConfig]:
    """Load session configuration from JSON."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return AudioConfig.from_dict(json.load(f))
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        log.warning("Failed to load config: %s", e)
        return None


def fmt_size(num_bytes: int) -> str:
    """Format byte count to human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1048576:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1048576:.2f} MB"


# ═══════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR – generate + save in one call
# ═══════════════════════════════════════════════════════════════════════
def generate_and_save(cfg: AudioConfig, pcb: ProgressCB = None,
                      export_flac: bool = False) -> str:
    """Generate audio from config and save to file. Returns saved path."""
    output_dir = cfg.output_dir or DEF_OUTDIR
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, cfg.auto_filename())

    log.info("=" * 56)
    log.info("  Mode         : %s", cfg.mode)
    log.info("  Beat freq    : %.2f Hz", cfg.beat_freq)
    log.info("  Carrier freq : %.2f Hz", cfg.carrier_freq)
    log.info("  Tone type    : %s", cfg.tone_type)
    log.info("  Duration     : %dm %ds (%.1f s total)",
             cfg.duration_min, cfg.duration_sec, cfg.total_seconds)
    log.info("  Sample rate  : %d Hz", cfg.sample_rate)
    log.info("  Volume       : %.2f", cfg.volume)
    log.info("  Noise        : %s @ %.0f%%",
             cfg.noise_type, cfg.noise_intensity * 100)
    log.info("  Fade         : %.2f s", cfg.fade_duration)
    log.info("  Est. size    : %s", fmt_size(cfg.est_wav_bytes))
    log.info("  Total samples: %s", f"{cfg.total_samples:,}")
    log.info("  Output       : %s", filepath)
    log.info("=" * 56)

    audio = generate_audio(cfg, pcb)
    saved_path = save_wav(audio, filepath, cfg.sample_rate)

    if export_flac:
        save_flac(audio, filepath, cfg.sample_rate)

    if pcb:
        pcb(1.0, f"Saved -> {saved_path}")
    return saved_path


def generate_all_modes(template: AudioConfig, pcb: ProgressCB = None,
                       export_flac: bool = False) -> List[str]:
    """Batch generate all predefined brainwave modes."""
    mode_names = list(BRAINWAVE_MODES)
    output_files: List[str] = []
    for i, name in enumerate(mode_names):
        cfg = AudioConfig.from_dict(template.to_dict())
        cfg.mode = name
        cfg.beat_freq = BRAINWAVE_MODES[name]["beat"]
        cfg.output_filename = ""
        if pcb:
            pcb(i / len(mode_names),
                f"[{i + 1}/{len(mode_names)}] {name}…")
        path = generate_and_save(cfg, export_flac=export_flac)
        output_files.append(path)
    if pcb:
        pcb(1.0, f"All {len(mode_names)} modes generated.")
    return output_files


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════
def build_cli_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the command-line interface."""
    ap = argparse.ArgumentParser(
        prog="brainwave_gen",
        description="Brainwave Audio Tone Generator – CLI & GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s --cli -m study --duration-min 10
              %(prog)s --cli -m custom --beat-freq 12 --tone-type isochronic
              %(prog)s --cli --generate-all --duration-min 5
              %(prog)s                          # launch GUI (default)
        """),
    )
    interface_group = ap.add_mutually_exclusive_group()
    interface_group.add_argument(
        "--cli", action="store_true", help="Run in command-line mode")
    interface_group.add_argument(
        "--gui", action="store_true", help="Run in graphical mode (default)")

    modes_list = list(BRAINWAVE_MODES) + ["custom"]
    ap.add_argument("-m", "--mode", choices=modes_list, default="study",
                    help="Brainwave mode preset or 'custom'")
    ap.add_argument("--duration-min", type=int, default=5,
                    help="Duration in minutes (default: 5)")
    ap.add_argument("--duration-sec", type=int, default=0,
                    help="Additional seconds (default: 0)")
    ap.add_argument("--carrier-freq", type=float, default=DEF_CARRIER,
                    help=f"Carrier/base frequency in Hz (default: {DEF_CARRIER})")
    ap.add_argument("--beat-freq", type=float, default=None,
                    help="Beat frequency in Hz (overrides mode default)")
    ap.add_argument("--volume", type=float, default=DEF_VOLUME,
                    help=f"Output volume 0.0–1.0 (default: {DEF_VOLUME})")
    ap.add_argument("--sample-rate", type=int, default=DEF_SR,
                    help=f"Sample rate in Hz (default: {DEF_SR})")
    ap.add_argument("--tone-type", choices=TONE_TYPES, default="binaural",
                    help="Type of tone generation (default: binaural)")
    ap.add_argument("--noise-type", choices=NOISE_TYPES, default="none",
                    help="Background noise type (default: none)")
    ap.add_argument("--noise-intensity", type=float, default=0.0,
                    help="Noise mix level 0.0–1.0 (default: 0.0)")
    ap.add_argument("--fade-duration", type=float, default=DEF_FADE,
                    help=f"Fade in/out seconds (default: {DEF_FADE})")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="Output filename (auto-generated if empty)")
    ap.add_argument("--output-dir", type=str, default=DEF_OUTDIR,
                    help=f"Output directory (default: {DEF_OUTDIR})")
    ap.add_argument("--generate-all", action="store_true",
                    help="Batch-generate every predefined mode")
    ap.add_argument("--flac", action="store_true",
                    help="Also export FLAC (requires soundfile)")
    ap.add_argument("--am", action="store_true",
                    help="Enable amplitude modulation")
    ap.add_argument("--am-freq", type=float, default=1.0,
                    help="AM frequency in Hz (default: 1.0)")
    ap.add_argument("--am-depth", type=float, default=0.5,
                    help="AM depth 0.0–1.0 (default: 0.5)")
    return ap


def run_cli(args: argparse.Namespace):
    """Execute generation based on parsed CLI arguments."""
    cfg = AudioConfig(
        mode=args.mode,
        carrier_freq=args.carrier_freq,
        tone_type=args.tone_type,
        duration_min=args.duration_min,
        duration_sec=args.duration_sec,
        volume=args.volume,
        sample_rate=args.sample_rate,
        noise_type=args.noise_type,
        noise_intensity=args.noise_intensity,
        fade_duration=args.fade_duration,
        output_filename=args.output,
        output_dir=args.output_dir,
        am_enabled=args.am,
        am_freq=args.am_freq,
        am_depth=args.am_depth,
    )

    # Resolve beat frequency from mode or explicit argument
    if args.mode == "custom":
        if args.beat_freq is None:
            sys.exit("Error: --beat-freq is required for custom mode.")
        cfg.beat_freq = args.beat_freq
    else:
        cfg.beat_freq = BRAINWAVE_MODES[args.mode]["beat"]

    # Explicit --beat-freq always overrides
    if args.beat_freq is not None:
        cfg.beat_freq = args.beat_freq

    # Validate
    errors = cfg.validate()
    if errors:
        print("Validation errors:")
        for err in errors:
            print(f"  x {err}")
        sys.exit(1)

    # Generate
    if args.generate_all:
        print(f"\n  Batch: generating all {len(BRAINWAVE_MODES)} modes...\n")
        files = generate_all_modes(cfg, export_flac=args.flac)
        print(f"\n  {len(files)} files written:")
        for f in files:
            print(f"   {f}")
    else:
        print(f"\n  Generating '{cfg.mode}' ...\n")
        path = generate_and_save(cfg, export_flac=args.flac)
        size_str = fmt_size(os.path.getsize(path))
        print(f"\n  Saved -> {path}  ({size_str})")


# ═══════════════════════════════════════════════════════════════════════
#  GUI (Tkinter + embedded Matplotlib preview)
# ═══════════════════════════════════════════════════════════════════════
def run_gui():
    """Launch the Tkinter graphical user interface."""
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # ─── Colour palettes for light and dark themes ──────────
    PALETTE_LIGHT = {
        "bg": "#f0f0f0", "fg": "#1a1a1a",
        "entry_bg": "#ffffff", "entry_fg": "#000000",
        "accent": "#3874c8", "frame_bg": "#e8e8e8",
        "plot_bg": "#ffffff", "plot_line": ("#3874c8", "#c0392b"),
        "plot_grid": "#cccccc", "plot_text": "#000000",
        "btn_bg": "#3874c8", "btn_fg": "#ffffff",
        "status_bg": "#e0e0e0", "status_fg": "#333333",
    }
    PALETTE_DARK = {
        "bg": "#2b2b2b", "fg": "#dcdcdc",
        "entry_bg": "#3c3c3c", "entry_fg": "#e0e0e0",
        "accent": "#5b9bd5", "frame_bg": "#333333",
        "plot_bg": "#2b2b2b", "plot_line": ("#5b9bd5", "#e07050"),
        "plot_grid": "#444444", "plot_text": "#d0d0d0",
        "btn_bg": "#5b9bd5", "btn_fg": "#ffffff",
        "status_bg": "#333333", "status_fg": "#cccccc",
    }

    class Application:
        """Main GUI application class."""

        def __init__(self, root: tk.Tk):
            self.root = root
            root.title("Brainwave Audio Tone Generator")
            root.minsize(1080, 780)
            root.protocol("WM_DELETE_WINDOW", self._on_quit)

            self._current_theme = "light"
            self._generating = False

            # ── Tkinter variables ───────────────────────────
            self.var_mode = tk.StringVar(value="study")
            self.var_beat = tk.StringVar(value="14.0")
            self.var_carrier = tk.StringVar(value=str(DEF_CARRIER))
            self.var_tone = tk.StringVar(value="binaural")
            self.var_dur_min = tk.StringVar(value="5")
            self.var_dur_sec = tk.StringVar(value="0")
            self.var_volume = tk.StringVar(value=str(DEF_VOLUME))
            self.var_sample_rate = tk.StringVar(value=str(DEF_SR))
            self.var_noise = tk.StringVar(value="none")
            self.var_noise_pct = tk.DoubleVar(value=0.0)
            self.var_fade = tk.StringVar(value=str(DEF_FADE))
            self.var_filename = tk.StringVar(value="")
            self.var_outdir = tk.StringVar(value=DEF_OUTDIR)
            self.var_am = tk.BooleanVar(value=False)
            self.var_am_freq = tk.StringVar(value="1.0")
            self.var_am_depth = tk.StringVar(value="0.5")
            self.var_flac = tk.BooleanVar(value=False)
            self.var_progress = tk.DoubleVar(value=0.0)
            self.var_status = tk.StringVar(value="Ready.")
            self.var_info = tk.StringVar(value="")

            self._build_ui()
            self._apply_theme()
            self._on_mode_changed()
            self._update_info()

        # ────────────────────────────────────────────────────
        #  UI CONSTRUCTION
        # ────────────────────────────────────────────────────
        def _build_ui(self):
            style = ttk.Style(self.root)
            style.theme_use("clam")

            # Main container
            container = ttk.Frame(self.root, padding=8)
            container.pack(fill="both", expand=True)

            # Two-column layout: controls left, preview right
            left_panel = ttk.Frame(container)
            right_panel = ttk.Frame(container)
            left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
            container.columnconfigure(0, weight=2, minsize=380)
            container.columnconfigure(1, weight=3, minsize=440)
            container.rowconfigure(0, weight=1)

            self._build_left_panel(left_panel)
            self._build_right_panel(right_panel)

            # Bottom bar
            bottom = ttk.Frame(container)
            bottom.grid(row=1, column=0, columnspan=2,
                        sticky="ew", pady=(8, 0))
            self._build_bottom_bar(bottom)

            # ── Variable traces for live info updates ───────
            watch_vars = [
                self.var_mode, self.var_beat, self.var_carrier,
                self.var_tone, self.var_dur_min, self.var_dur_sec,
                self.var_volume, self.var_sample_rate, self.var_noise,
                self.var_fade,
            ]
            for var in watch_vars:
                var.trace_add("write", lambda *_: self._update_info())

            self.var_noise_pct.trace_add(
                "write", lambda *_: self._update_info())
            self.var_noise_pct.trace_add(
                "write", lambda *_: self._noise_label.configure(
                    text=f"{self.var_noise_pct.get() * 100:.0f}%"))

        def _build_left_panel(self, parent: ttk.Frame):
            """Build the left side controls panel."""
            parent.columnconfigure(0, weight=1)
            row = 0

            # ── Mode & Frequency ────────────────────────────
            frame_mode = ttk.LabelFrame(
                parent, text=" Mode & Frequency ", padding=6)
            frame_mode.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            frame_mode.columnconfigure(1, weight=1)
            row += 1

            ttk.Label(frame_mode, text="Mode:").grid(
                row=0, column=0, sticky="w", padx=2, pady=2)
            modes = list(BRAINWAVE_MODES) + ["custom"]
            self._mode_combo = ttk.Combobox(
                frame_mode, textvariable=self.var_mode,
                values=modes, state="readonly", width=18)
            self._mode_combo.grid(
                row=0, column=1, sticky="ew", padx=2, pady=2)
            self.var_mode.trace_add(
                "write", lambda *_: self._on_mode_changed())

            ttk.Label(frame_mode, text="Beat Hz:").grid(
                row=1, column=0, sticky="w", padx=2, pady=2)
            self._beat_entry = ttk.Entry(
                frame_mode, textvariable=self.var_beat, width=12)
            self._beat_entry.grid(
                row=1, column=1, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_mode, text="Carrier Hz:").grid(
                row=2, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(
                frame_mode, textvariable=self.var_carrier, width=12
            ).grid(row=2, column=1, sticky="ew", padx=2, pady=2)

            # Mode description label
            self._mode_desc = ttk.Label(
                frame_mode, text="", foreground="#666666")
            self._mode_desc.grid(
                row=3, column=0, columnspan=2, sticky="w", padx=2, pady=2)

            # ── Tone Type ───────────────────────────────────
            frame_tone = ttk.LabelFrame(
                parent, text=" Tone Type ", padding=6)
            frame_tone.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            row += 1

            for i, tone in enumerate(TONE_TYPES):
                ttk.Radiobutton(
                    frame_tone, text=tone.capitalize(),
                    variable=self.var_tone, value=tone
                ).grid(row=0, column=i, padx=6, pady=2)

            # ── Duration & Audio Settings ───────────────────
            frame_dur = ttk.LabelFrame(
                parent, text=" Duration & Audio ", padding=6)
            frame_dur.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            frame_dur.columnconfigure(1, weight=1)
            frame_dur.columnconfigure(3, weight=1)
            row += 1

            ttk.Label(frame_dur, text="Min:").grid(
                row=0, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(
                frame_dur, textvariable=self.var_dur_min, width=6
            ).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_dur, text="Sec:").grid(
                row=0, column=2, sticky="w", padx=(10, 2), pady=2)
            ttk.Entry(
                frame_dur, textvariable=self.var_dur_sec, width=6
            ).grid(row=0, column=3, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_dur, text="Volume:").grid(
                row=1, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(
                frame_dur, textvariable=self.var_volume, width=6
            ).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_dur, text="SR:").grid(
                row=1, column=2, sticky="w", padx=(10, 2), pady=2)
            sr_combo = ttk.Combobox(
                frame_dur, textvariable=self.var_sample_rate,
                values=["22050", "44100", "48000", "96000"],
                width=8)
            sr_combo.grid(row=1, column=3, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_dur, text="Fade (s):").grid(
                row=2, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(
                frame_dur, textvariable=self.var_fade, width=6
            ).grid(row=2, column=1, sticky="ew", padx=2, pady=2)

            # ── Noise ──────────────────────────────────────
            frame_noise = ttk.LabelFrame(
                parent, text=" Background Noise ", padding=6)
            frame_noise.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            frame_noise.columnconfigure(1, weight=1)
            row += 1

            ttk.Label(frame_noise, text="Type:").grid(
                row=0, column=0, sticky="w", padx=2, pady=2)
            ttk.Combobox(
                frame_noise, textvariable=self.var_noise,
                values=list(NOISE_TYPES), state="readonly", width=10
            ).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_noise, text="Intensity:").grid(
                row=1, column=0, sticky="w", padx=2, pady=2)
            slider_frame = ttk.Frame(frame_noise)
            slider_frame.grid(
                row=1, column=1, sticky="ew", padx=2, pady=2)
            slider_frame.columnconfigure(0, weight=1)

            ttk.Scale(
                slider_frame, from_=0.0, to=1.0,
                variable=self.var_noise_pct, orient="horizontal"
            ).grid(row=0, column=0, sticky="ew")
            self._noise_label = ttk.Label(slider_frame, text="0%", width=5)
            self._noise_label.grid(row=0, column=1, padx=(4, 0))

            # ── AM / Output / Options ───────────────────────
            frame_opts = ttk.LabelFrame(
                parent, text=" Options ", padding=6)
            frame_opts.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            frame_opts.columnconfigure(1, weight=1)
            row += 1

            ttk.Checkbutton(
                frame_opts, text="Amplitude Modulation",
                variable=self.var_am
            ).grid(row=0, column=0, columnspan=2, sticky="w", pady=1)

            ttk.Label(frame_opts, text="AM Hz:").grid(
                row=1, column=0, sticky="w", padx=2, pady=1)
            ttk.Entry(
                frame_opts, textvariable=self.var_am_freq, width=8
            ).grid(row=1, column=1, sticky="w", padx=2, pady=1)

            ttk.Label(frame_opts, text="AM Depth:").grid(
                row=2, column=0, sticky="w", padx=2, pady=1)
            ttk.Entry(
                frame_opts, textvariable=self.var_am_depth, width=8
            ).grid(row=2, column=1, sticky="w", padx=2, pady=1)

            if HAS_SOUNDFILE:
                ttk.Checkbutton(
                    frame_opts, text="Also export FLAC",
                    variable=self.var_flac
                ).grid(row=3, column=0, columnspan=2, sticky="w", pady=1)

            # ── Output Path ─────────────────────────────────
            frame_out = ttk.LabelFrame(
                parent, text=" Output ", padding=6)
            frame_out.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            frame_out.columnconfigure(1, weight=1)
            row += 1

            ttk.Label(frame_out, text="Filename:").grid(
                row=0, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(
                frame_out, textvariable=self.var_filename, width=24
            ).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

            ttk.Label(frame_out, text="Directory:").grid(
                row=1, column=0, sticky="w", padx=2, pady=2)
            dir_frame = ttk.Frame(frame_out)
            dir_frame.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
            dir_frame.columnconfigure(0, weight=1)
            ttk.Entry(
                dir_frame, textvariable=self.var_outdir
            ).grid(row=0, column=0, sticky="ew")
            ttk.Button(
                dir_frame, text="Browse", width=7,
                command=self._browse_dir
            ).grid(row=0, column=1, padx=(4, 0))

        def _build_right_panel(self, parent: ttk.Frame):
            """Build the right side: info, preview, buttons."""
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(2, weight=1)
            row = 0

            # ── Info / Summary ──────────────────────────────
            frame_info = ttk.LabelFrame(
                parent, text=" Audio Info ", padding=6)
            frame_info.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            frame_info.columnconfigure(0, weight=1)
            row += 1

            self._info_label = ttk.Label(
                frame_info, textvariable=self.var_info,
                justify="left", wraplength=420)
            self._info_label.grid(row=0, column=0, sticky="w")

            # ── Buttons ─────────────────────────────────────
            frame_btns = ttk.Frame(parent, padding=4)
            frame_btns.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            row += 1

            self._btn_generate = ttk.Button(
                frame_btns, text="Generate",
                command=self._on_generate)
            self._btn_generate.pack(side="left", padx=4)

            self._btn_gen_all = ttk.Button(
                frame_btns, text="Generate All Modes",
                command=self._on_generate_all)
            self._btn_gen_all.pack(side="left", padx=4)

            ttk.Button(
                frame_btns, text="Preview",
                command=self._on_preview
            ).pack(side="left", padx=4)

            ttk.Separator(frame_btns, orient="vertical").pack(
                side="left", fill="y", padx=8)

            ttk.Button(
                frame_btns, text="Save Config",
                command=self._on_save_config
            ).pack(side="left", padx=4)

            ttk.Button(
                frame_btns, text="Load Config",
                command=self._on_load_config
            ).pack(side="left", padx=4)

            ttk.Separator(frame_btns, orient="vertical").pack(
                side="left", fill="y", padx=8)

            self._theme_btn = ttk.Button(
                frame_btns, text="Dark Theme",
                command=self._toggle_theme)
            self._theme_btn.pack(side="left", padx=4)

            # ── Waveform Preview ────────────────────────────
            frame_plot = ttk.LabelFrame(
                parent, text=" Waveform Preview (first 2 s) ", padding=4)
            frame_plot.grid(
                row=row, column=0, sticky="nsew", pady=(0, 4))
            frame_plot.columnconfigure(0, weight=1)
            frame_plot.rowconfigure(0, weight=1)
            row += 1

            self._fig = Figure(figsize=(5, 3), dpi=90)
            self._fig.subplots_adjust(
                left=0.08, right=0.97, top=0.92, bottom=0.15)
            self._ax = self._fig.add_subplot(111)
            self._ax.set_xlabel("Time (s)")
            self._ax.set_ylabel("Amplitude")
            self._ax.set_title("No preview yet")
            self._ax.grid(True, alpha=0.3)

            self._canvas = FigureCanvasTkAgg(self._fig, master=frame_plot)
            self._canvas.get_tk_widget().grid(
                row=0, column=0, sticky="nsew")
            self._canvas.draw()

        def _build_bottom_bar(self, parent: ttk.Frame):
            """Build progress bar and status area."""
            parent.columnconfigure(1, weight=1)

            self._progress_bar = ttk.Progressbar(
                parent, variable=self.var_progress,
                maximum=1.0, length=300, mode="determinate")
            self._progress_bar.grid(
                row=0, column=0, sticky="ew", padx=(0, 8))

            self._status_label = ttk.Label(
                parent, textvariable=self.var_status, anchor="w")
            self._status_label.grid(row=0, column=1, sticky="ew")

        # ────────────────────────────────────────────────────
        #  THEME MANAGEMENT
        # ────────────────────────────────────────────────────
        def _apply_theme(self):
            pal = (PALETTE_LIGHT if self._current_theme == "light"
                   else PALETTE_DARK)
            style = ttk.Style()

            self.root.configure(bg=pal["bg"])

            style.configure(".", background=pal["bg"],
                            foreground=pal["fg"])
            style.configure("TFrame", background=pal["bg"])
            style.configure("TLabel", background=pal["bg"],
                            foreground=pal["fg"])
            style.configure("TLabelframe", background=pal["bg"],
                            foreground=pal["fg"])
            style.configure("TLabelframe.Label", background=pal["bg"],
                            foreground=pal["accent"])
            style.configure("TButton", background=pal["frame_bg"],
                            foreground=pal["fg"])
            style.configure("TRadiobutton", background=pal["bg"],
                            foreground=pal["fg"])
            style.configure("TCheckbutton", background=pal["bg"],
                            foreground=pal["fg"])
            style.configure("TCombobox",
                            fieldbackground=pal["entry_bg"],
                            foreground=pal["entry_fg"],
                            background=pal["frame_bg"])
            style.configure("TEntry",
                            fieldbackground=pal["entry_bg"],
                            foreground=pal["entry_fg"])
            style.configure("Horizontal.TScale",
                            background=pal["bg"],
                            troughcolor=pal["frame_bg"])
            style.configure("Horizontal.TProgressbar",
                            background=pal["accent"],
                            troughcolor=pal["frame_bg"])
            style.configure("TSeparator", background=pal["frame_bg"])

            # Matplotlib colours
            self._fig.set_facecolor(pal["plot_bg"])
            self._ax.set_facecolor(pal["plot_bg"])
            for spine in self._ax.spines.values():
                spine.set_color(pal["plot_grid"])
            self._ax.tick_params(colors=pal["plot_text"])
            self._ax.xaxis.label.set_color(pal["plot_text"])
            self._ax.yaxis.label.set_color(pal["plot_text"])
            self._ax.title.set_color(pal["plot_text"])
            self._ax.grid(True, color=pal["plot_grid"], alpha=0.3)
            self._canvas.draw_idle()

        def _toggle_theme(self):
            if self._current_theme == "light":
                self._current_theme = "dark"
                self._theme_btn.configure(text="Light Theme")
            else:
                self._current_theme = "light"
                self._theme_btn.configure(text="Dark Theme")
            self._apply_theme()

        # ────────────────────────────────────────────────────
        #  EVENT HANDLERS
        # ────────────────────────────────────────────────────
        def _on_mode_changed(self, *_):
            """Update beat frequency and description when mode changes."""
            mode = self.var_mode.get()
            if mode in BRAINWAVE_MODES:
                info = BRAINWAVE_MODES[mode]
                self.var_beat.set(str(info["beat"]))
                self._beat_entry.configure(state="disabled")
                self._mode_desc.configure(text=info["label"])
            else:
                self._beat_entry.configure(state="normal")
                self._mode_desc.configure(text="Custom – enter beat Hz")

        def _update_info(self, *_):
            """Recalculate and display audio info summary."""
            try:
                cfg = self._build_config()
                errors = cfg.validate()
                if errors:
                    self.var_info.set("  ".join(errors))
                    return
                lines = [
                    f"Mode: {cfg.mode}  |  Tone: {cfg.tone_type}",
                    f"Beat: {cfg.beat_freq:.1f} Hz  |  "
                    f"Carrier: {cfg.carrier_freq:.1f} Hz",
                    f"Duration: {cfg.duration_min}m {cfg.duration_sec}s  "
                    f"({cfg.total_seconds:.0f}s)",
                    f"Samples: {cfg.total_samples:,}  |  "
                    f"SR: {cfg.sample_rate} Hz",
                    f"Volume: {cfg.volume:.2f}  |  "
                    f"Fade: {cfg.fade_duration:.2f}s",
                    f"Noise: {cfg.noise_type}  |  "
                    f"Est. size: {fmt_size(cfg.est_wav_bytes)}",
                ]
                if cfg.tone_type == "binaural":
                    lf = cfg.carrier_freq - cfg.beat_freq / 2
                    rf = cfg.carrier_freq + cfg.beat_freq / 2
                    lines.append(
                        f"L-channel: {lf:.1f} Hz  |  "
                        f"R-channel: {rf:.1f} Hz")
                self.var_info.set("\n".join(lines))
            except (ValueError, TypeError):
                self.var_info.set("Enter valid numeric values.")

        def _build_config(self) -> AudioConfig:
            """Build AudioConfig from current GUI variable values."""
            mode = self.var_mode.get()
            beat = float(self.var_beat.get() or "0")
            if mode in BRAINWAVE_MODES:
                beat = BRAINWAVE_MODES[mode]["beat"]

            return AudioConfig(
                mode=mode,
                beat_freq=beat,
                carrier_freq=float(self.var_carrier.get() or "200"),
                tone_type=self.var_tone.get(),
                duration_min=int(self.var_dur_min.get() or "0"),
                duration_sec=int(self.var_dur_sec.get() or "0"),
                volume=float(self.var_volume.get() or "0.7"),
                sample_rate=int(self.var_sample_rate.get() or "44100"),
                noise_type=self.var_noise.get(),
                noise_intensity=self.var_noise_pct.get(),
                fade_duration=float(self.var_fade.get() or "0.5"),
                output_filename=self.var_filename.get(),
                output_dir=self.var_outdir.get(),
                am_enabled=self.var_am.get(),
                am_freq=float(self.var_am_freq.get() or "1.0"),
                am_depth=float(self.var_am_depth.get() or "0.5"),
            )

        def _browse_dir(self):
            d = filedialog.askdirectory(
                initialdir=self.var_outdir.get(),
                title="Select Output Directory")
            if d:
                self.var_outdir.set(d)

        def _progress_callback(self, value: float, message: str):
            """Thread-safe progress update."""
            self.root.after(0, self._update_progress, value, message)

        def _update_progress(self, value: float, message: str):
            self.var_progress.set(value)
            self.var_status.set(message)

        def _set_busy(self, busy: bool):
            self._generating = busy
            state = "disabled" if busy else "normal"
            self._btn_generate.configure(state=state)
            self._btn_gen_all.configure(state=state)

        # ── Generate ────────────────────────────────────────
        def _on_generate(self):
            if self._generating:
                return
            try:
                cfg = self._build_config()
            except (ValueError, TypeError) as e:
                messagebox.showerror("Input Error", str(e))
                return

            errors = cfg.validate()
            if errors:
                messagebox.showerror(
                    "Validation Error", "\n".join(errors))
                return

            self._set_busy(True)
            self.var_progress.set(0.0)
            self.var_status.set("Generating…")

            def _worker():
                try:
                    path = generate_and_save(
                        cfg, pcb=self._progress_callback,
                        export_flac=self.var_flac.get())
                    save_config(cfg)
                    size = fmt_size(os.path.getsize(path))
                    self.root.after(0, lambda: (
                        self.var_status.set(
                            f"Saved: {path} ({size})"),
                        self.var_progress.set(1.0),
                        messagebox.showinfo(
                            "Success",
                            f"Audio saved to:\n{path}\nSize: {size}"),
                    ))
                except Exception as e:
                    log.exception("Generation failed")
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", str(e)))
                finally:
                    self.root.after(0, lambda: self._set_busy(False))

            threading.Thread(target=_worker, daemon=True).start()

        # ── Generate All ────────────────────────────────────
        def _on_generate_all(self):
            if self._generating:
                return
            try:
                cfg = self._build_config()
            except (ValueError, TypeError) as e:
                messagebox.showerror("Input Error", str(e))
                return

            self._set_busy(True)
            self.var_progress.set(0.0)
            self.var_status.set("Batch generating all modes…")

            def _worker():
                try:
                    files = generate_all_modes(
                        cfg, pcb=self._progress_callback,
                        export_flac=self.var_flac.get())
                    save_config(cfg)
                    self.root.after(0, lambda: (
                        self.var_status.set(
                            f"All {len(files)} modes generated."),
                        self.var_progress.set(1.0),
                        messagebox.showinfo(
                            "Batch Complete",
                            f"{len(files)} files saved to:\n"
                            f"{cfg.output_dir or DEF_OUTDIR}"),
                    ))
                except Exception as e:
                    log.exception("Batch generation failed")
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", str(e)))
                finally:
                    self.root.after(0, lambda: self._set_busy(False))

            threading.Thread(target=_worker, daemon=True).start()

        # ── Preview ─────────────────────────────────────────
        def _on_preview(self):
            """Generate and display a 2-second waveform preview."""
            try:
                cfg = self._build_config()
            except (ValueError, TypeError) as e:
                messagebox.showerror("Input Error", str(e))
                return

            errors = cfg.validate()
            if errors:
                messagebox.showerror(
                    "Validation Error", "\n".join(errors))
                return

            self.var_status.set("Rendering preview…")

            # Build a short 2-second config for preview
            preview_cfg = AudioConfig.from_dict(cfg.to_dict())
            preview_cfg.duration_min = 0
            preview_cfg.duration_sec = 2
            preview_cfg.fade_duration = min(
                cfg.fade_duration, 0.3)

            try:
                audio = generate_audio(preview_cfg)
            except Exception as e:
                messagebox.showerror("Preview Error", str(e))
                return

            # Plot
            pal = (PALETTE_LIGHT if self._current_theme == "light"
                   else PALETTE_DARK)
            sr = preview_cfg.sample_rate
            t = np.arange(audio.shape[0]) / sr

            self._ax.clear()
            self._ax.set_facecolor(pal["plot_bg"])
            self._ax.plot(
                t, audio[:, 0],
                color=pal["plot_line"][0],
                linewidth=0.5, alpha=0.85, label="Left")
            if audio.shape[1] > 1:
                self._ax.plot(
                    t, audio[:, 1],
                    color=pal["plot_line"][1],
                    linewidth=0.5, alpha=0.65, label="Right")
            self._ax.set_xlabel("Time (s)")
            self._ax.set_ylabel("Amplitude")
            self._ax.set_title(
                f"Preview: {cfg.mode} – {cfg.tone_type} "
                f"({cfg.beat_freq:.1f} Hz)")
            self._ax.set_xlim(0, 2)
            self._ax.set_ylim(-1.05, 1.05)
            self._ax.legend(loc="upper right", fontsize=8)
            self._ax.grid(True, color=pal["plot_grid"], alpha=0.3)

            # Recolour text elements for current theme
            self._ax.title.set_color(pal["plot_text"])
            self._ax.xaxis.label.set_color(pal["plot_text"])
            self._ax.yaxis.label.set_color(pal["plot_text"])
            self._ax.tick_params(colors=pal["plot_text"])
            for spine in self._ax.spines.values():
                spine.set_color(pal["plot_grid"])

            self._canvas.draw()
            self.var_status.set("Preview rendered.")

        # ── Config Save / Load ──────────────────────────────
        def _on_save_config(self):
            try:
                cfg = self._build_config()
                path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON", "*.json")],
                    initialfile=CFG_FILE,
                    title="Save Configuration")
                if path:
                    save_config(cfg, path)
                    self.var_status.set(f"Config saved: {path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def _on_load_config(self):
            path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                title="Load Configuration")
            if not path:
                return
            cfg = load_config(path)
            if cfg is None:
                messagebox.showwarning(
                    "Load Failed", "Could not parse config file.")
                return
            self._apply_config_to_ui(cfg)
            self.var_status.set(f"Config loaded: {path}")

        def _apply_config_to_ui(self, cfg: AudioConfig):
            """Push an AudioConfig's values into the GUI variables."""
            self.var_mode.set(cfg.mode)
            self.var_beat.set(str(cfg.beat_freq))
            self.var_carrier.set(str(cfg.carrier_freq))
            self.var_tone.set(cfg.tone_type)
            self.var_dur_min.set(str(cfg.duration_min))
            self.var_dur_sec.set(str(cfg.duration_sec))
            self.var_volume.set(str(cfg.volume))
            self.var_sample_rate.set(str(cfg.sample_rate))
            self.var_noise.set(cfg.noise_type)
            self.var_noise_pct.set(cfg.noise_intensity)
            self.var_fade.set(str(cfg.fade_duration))
            self.var_filename.set(cfg.output_filename)
            self.var_outdir.set(cfg.output_dir)
            self.var_am.set(cfg.am_enabled)
            self.var_am_freq.set(str(cfg.am_freq))
            self.var_am_depth.set(str(cfg.am_depth))

        # ── Quit ────────────────────────────────────────────
        def _on_quit(self):
            if self._generating:
                if not messagebox.askyesno(
                        "Quit?",
                        "Audio is being generated. Quit anyway?"):
                    return
            self.root.destroy()

    # ── Launch the Tk main loop ─────────────────────────────
    root = tk.Tk()
    app = Application(root)

    # Try loading last session config
    last_cfg = load_config()
    if last_cfg is not None:
        app._apply_config_to_ui(last_cfg)
        log.info("Loaded previous session config.")

    root.mainloop()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()