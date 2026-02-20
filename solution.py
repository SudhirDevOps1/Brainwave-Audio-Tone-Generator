
"""
🧠 Brainwave Audio Tone Generator - Professional Edition
=========================================================
A beautiful, responsive, production-level application for generating
brainwave entrainment audio tones with stunning UI and comprehensive features.

Supports binaural beats, monaural beats, isochronic tones, and pure sine waves.
Background noise mixing, waveform preview, batch generation, and more.

Dependencies: numpy, scipy, matplotlib
Optional: soundfile (for FLAC export)

Usage:
    CLI Mode:  python solution.py cli --mode relax --duration 300 --volume 0.7
    GUI Mode:  python solution.py gui
    Default:   python solution.py  (launches GUI)
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, List, Callable
from enum import Enum

import numpy as np
from scipy.io import wavfile

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BrainwaveGenerator")


# ============================================================
# DATA MODELS & ENUMERATIONS
# ============================================================

class ToneType(Enum):
    """Supported tone generation types."""
    PURE = "pure"
    BINAURAL = "binaural"
    MONAURAL = "monaural"
    ISOCHRONIC = "isochronic"


class NoiseType(Enum):
    """Supported background noise types."""
    NONE = "none"
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"


@dataclass
class BrainwaveMode:
    """Definition of a predefined brainwave entrainment mode."""
    name: str
    beat_frequency: float
    description: str
    band: str
    emoji: str
    color: str  # hex color for UI


# Predefined brainwave modes with colors and emojis
BRAINWAVE_MODES: Dict[str, BrainwaveMode] = {
    "reading":       BrainwaveMode("reading", 10.0, "10 Hz Alpha - Reading Enhancement", "alpha", "📖", "#4FC3F7"),
    "study":         BrainwaveMode("study", 14.0, "14 Hz Low Beta - Study Aid", "beta", "📚", "#7986CB"),
    "deep_focus":    BrainwaveMode("deep_focus", 16.0, "16 Hz Beta - Deep Focus", "beta", "🎯", "#FF7043"),
    "gamma_focus":   BrainwaveMode("gamma_focus", 40.0, "40 Hz Gamma - Peak Focus", "gamma", "⚡", "#FFD54F"),
    "relax":         BrainwaveMode("relax", 8.0, "8 Hz Alpha - Relaxation", "alpha", "🌊", "#4DB6AC"),
    "stress_relief": BrainwaveMode("stress_relief", 6.0, "6 Hz Theta - Stress Relief", "theta", "🧘", "#AED581"),
    "sleep":         BrainwaveMode("sleep", 2.0, "2 Hz Delta - Sleep Induction", "delta", "😴", "#9575CD"),
    "meditation":    BrainwaveMode("meditation", 7.0, "7 Hz Theta - Meditation", "theta", "🕉️", "#F48FB1"),
    "creativity":    BrainwaveMode("creativity", 5.0, "5 Hz Theta - Creativity Boost", "theta", "🎨", "#FFB74D"),
    "memory_boost":  BrainwaveMode("memory_boost", 18.0, "18 Hz Beta - Memory Enhancement", "beta", "🧠", "#E57373"),
}

# Band color mapping
BAND_COLORS = {
    "delta": "#9575CD",
    "theta": "#AED581",
    "alpha": "#4FC3F7",
    "beta":  "#FF7043",
    "gamma": "#FFD54F",
}


@dataclass
class GenerationConfig:
    """Complete configuration for audio generation."""
    mode: str = "relax"
    beat_frequency: float = 8.0
    carrier_frequency: float = 200.0
    tone_type: str = "binaural"
    duration_seconds: float = 300.0
    volume: float = 0.7
    sample_rate: int = 44100
    noise_type: str = "none"
    noise_intensity: float = 0.3
    fade_duration: float = 2.0
    output_filename: str = ""
    output_directory: str = "generated_audio"
    export_flac: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def validate(self) -> List[str]:
        """Validate configuration and return list of error messages."""
        errors = []
        if self.beat_frequency < 0.5 or self.beat_frequency > 100:
            errors.append("Beat frequency must be between 0.5 and 100 Hz.")
        if self.carrier_frequency < 20 or self.carrier_frequency > 1000:
            errors.append("Carrier frequency must be between 20 and 1000 Hz.")
        if self.duration_seconds < 1 or self.duration_seconds > 7200:
            errors.append("Duration must be between 1 second and 2 hours (7200s).")
        if self.volume < 0.0 or self.volume > 1.0:
            errors.append("Volume must be between 0.0 and 1.0.")
        if self.sample_rate not in (22050, 44100, 48000, 96000):
            errors.append("Sample rate must be 22050, 44100, 48000, or 96000.")
        if self.noise_intensity < 0.0 or self.noise_intensity > 1.0:
            errors.append("Noise intensity must be between 0.0 and 1.0.")
        if self.fade_duration < 0.0 or self.fade_duration > self.duration_seconds / 2:
            errors.append("Fade duration must be >= 0 and <= half the total duration.")
        if self.tone_type not in [t.value for t in ToneType]:
            errors.append(f"Invalid tone type: {self.tone_type}")
        if self.noise_type not in [n.value for n in NoiseType]:
            errors.append(f"Invalid noise type: {self.noise_type}")
        return errors


# ============================================================
# DSP / AUDIO GENERATION ENGINE
# ============================================================

class AudioEngine:
    """
    Core digital signal processing engine for brainwave audio generation.
    All methods are stateless and operate on numpy arrays.
    """

    @staticmethod
    def generate_sine(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
        """
        Generate a pure sine wave.
        f(t) = sin(2π × frequency × t)
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return np.sin(2.0 * np.pi * frequency * t)

    @staticmethod
    def generate_binaural(
        carrier_freq: float, beat_freq: float,
        duration: float, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binaural beat as stereo (left, right) channels.
        Left ear:  carrier_freq Hz
        Right ear: carrier_freq + beat_freq Hz
        Brain perceives the difference frequency as a rhythmic beat.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        left = np.sin(2.0 * np.pi * carrier_freq * t)
        right = np.sin(2.0 * np.pi * (carrier_freq + beat_freq) * t)
        return left, right

    @staticmethod
    def generate_monaural(
        carrier_freq: float, beat_freq: float,
        duration: float, sample_rate: int
    ) -> np.ndarray:
        """
        Generate monaural beat (amplitude-modulated sine wave).
        f(t) = sin(2π·f_carrier·t) × (0.5 + 0.5·sin(2π·f_beat·t))
        No headphones required — the modulation is in the audio itself.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        carrier = np.sin(2.0 * np.pi * carrier_freq * t)
        modulator = 0.5 + 0.5 * np.sin(2.0 * np.pi * beat_freq * t)
        return carrier * modulator

    @staticmethod
    def generate_isochronic(
        carrier_freq: float, beat_freq: float,
        duration: float, sample_rate: int
    ) -> np.ndarray:
        """
        Generate isochronic tone (sharp on/off pulsing).
        Uses a sigmoid-smoothed square wave envelope to avoid harsh clicks.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        carrier = np.sin(2.0 * np.pi * carrier_freq * t)
        pulse_wave = np.sin(2.0 * np.pi * beat_freq * t)
        steepness = 10.0
        envelope = 1.0 / (1.0 + np.exp(-steepness * pulse_wave))
        return carrier * envelope

    @staticmethod
    def generate_white_noise(num_samples: int) -> np.ndarray:
        """White noise: equal energy at all frequencies."""
        return np.random.randn(num_samples)

    @staticmethod
    def generate_pink_noise(num_samples: int) -> np.ndarray:
        """
        Pink noise (1/f noise): energy decreases by 3dB per octave.
        Uses spectral shaping: apply 1/√f filter in frequency domain.
        """
        white = np.fft.rfft(np.random.randn(num_samples))
        freqs = np.fft.rfftfreq(num_samples)
        freqs[0] = 1.0
        pink_filter = 1.0 / np.sqrt(freqs)
        pink_filter[0] = 0.0
        pink = np.fft.irfft(white * pink_filter, n=num_samples)
        return pink

    @staticmethod
    def generate_brown_noise(num_samples: int) -> np.ndarray:
        """
        Brown (Brownian/red) noise: energy decreases by 6dB per octave.
        Generated by integrating white noise (cumulative sum → random walk).
        """
        white = np.random.randn(num_samples)
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)
        return brown

    @staticmethod
    def apply_fade(audio: np.ndarray, fade_duration: float, sample_rate: int) -> np.ndarray:
        """
        Apply smooth raised-cosine (Hann) fade-in and fade-out.
        Prevents click/pop artifacts at start and end.
        """
        fade_samples = int(fade_duration * sample_rate)
        if fade_samples == 0 or fade_samples * 2 > len(audio):
            return audio
        audio = audio.copy()
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))
        if audio.ndim == 1:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        elif audio.ndim == 2:
            for ch in range(audio.shape[1]):
                audio[:fade_samples, ch] *= fade_in
                audio[-fade_samples:, ch] *= fade_out
        return audio

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1.0, 1.0] range."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio

    @staticmethod
    def apply_limiter(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Soft limiter using tanh to prevent distortion."""
        return threshold * np.tanh(audio / threshold)

    @staticmethod
    def to_16bit_pcm(audio: np.ndarray) -> np.ndarray:
        """Convert float audio [-1,1] to 16-bit PCM integers."""
        audio_clipped = np.clip(audio, -1.0, 1.0)
        return (audio_clipped * 32767).astype(np.int16)


# ============================================================
# HIGH-LEVEL AUDIO GENERATOR
# ============================================================

class BrainwaveGenerator:
    """Orchestrates AudioEngine to produce complete audio files."""

    def __init__(self):
        self.engine = AudioEngine()

    def generate(
        self, config: GenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """Generate audio based on configuration. Returns (int16 array, sample_rate)."""

        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct:.0f}%] {msg}")

        beat_freq = config.beat_frequency
        if config.mode != "custom" and config.mode in BRAINWAVE_MODES:
            beat_freq = BRAINWAVE_MODES[config.mode].beat_frequency

        sr = config.sample_rate
        dur = config.duration_seconds
        carrier = config.carrier_frequency
        num_samples = int(sr * dur)

        report(5, f"Generating {config.tone_type} tone: carrier={carrier}Hz, beat={beat_freq}Hz, duration={dur}s")

        # Step 1: Generate base tone
        tone_type = ToneType(config.tone_type)
        if tone_type == ToneType.PURE:
            mono = self.engine.generate_sine(carrier, dur, sr)
            audio = np.column_stack([mono, mono])
            report(30, "Pure sine wave generated.")
        elif tone_type == ToneType.BINAURAL:
            left, right = self.engine.generate_binaural(carrier, beat_freq, dur, sr)
            audio = np.column_stack([left, right])
            report(30, f"Binaural beat: L={carrier}Hz, R={carrier + beat_freq}Hz")
        elif tone_type == ToneType.MONAURAL:
            mono = self.engine.generate_monaural(carrier, beat_freq, dur, sr)
            audio = np.column_stack([mono, mono])
            report(30, "Monaural beat generated.")
        elif tone_type == ToneType.ISOCHRONIC:
            mono = self.engine.generate_isochronic(carrier, beat_freq, dur, sr)
            audio = np.column_stack([mono, mono])
            report(30, "Isochronic tone generated.")
        else:
            raise ValueError(f"Unknown tone type: {config.tone_type}")

        # Step 2: Mix background noise
        noise_type = NoiseType(config.noise_type)
        if noise_type != NoiseType.NONE and config.noise_intensity > 0:
            report(45, f"Generating {noise_type.value} noise at {config.noise_intensity * 100:.0f}%...")
            if noise_type == NoiseType.WHITE:
                noise = self.engine.generate_white_noise(num_samples)
            elif noise_type == NoiseType.PINK:
                noise = self.engine.generate_pink_noise(num_samples)
            elif noise_type == NoiseType.BROWN:
                noise = self.engine.generate_brown_noise(num_samples)
            else:
                noise = np.zeros(num_samples)

            noise = self.engine.normalize(noise)
            nl = config.noise_intensity
            tl = 1.0 - nl * 0.5
            noise_stereo = np.column_stack([noise, noise])
            audio = tl * audio + nl * noise_stereo
            report(60, "Noise mixed into audio.")
        else:
            report(60, "No background noise.")

        # Step 3: Volume
        audio = audio * config.volume
        report(70, f"Volume applied: {config.volume}")

        # Step 4: Normalize
        audio = self.engine.normalize(audio)
        report(75, "Audio normalized.")

        # Step 5: Limiter
        audio = self.engine.apply_limiter(audio, threshold=0.95)
        report(80, "Limiter applied.")

        # Step 6: Fade
        if config.fade_duration > 0:
            audio = self.engine.apply_fade(audio, config.fade_duration, sr)
            report(85, f"Fade applied: {config.fade_duration}s in/out.")

        # Step 7: Convert to 16-bit PCM
        audio = self.engine.normalize(audio) * config.volume
        audio_pcm = self.engine.to_16bit_pcm(audio)
        report(95, "Converted to 16-bit PCM.")

        return audio_pcm, sr

    def save_wav(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> str:
        """Save as WAV file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        wavfile.write(filepath, sample_rate, audio_data)
        file_size = os.path.getsize(filepath)
        logger.info(f"WAV saved: {filepath} ({file_size / 1024 / 1024:.2f} MB)")
        return filepath

    def save_flac(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> Optional[str]:
        """Save as FLAC file (requires soundfile library)."""
        try:
            import soundfile as sf
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            audio_float = audio_data.astype(np.float32) / 32767.0
            sf.write(filepath, audio_float, sample_rate, format="FLAC")
            file_size = os.path.getsize(filepath)
            logger.info(f"FLAC saved: {filepath} ({file_size / 1024 / 1024:.2f} MB)")
            return filepath
        except ImportError:
            logger.warning("soundfile library not available. FLAC export skipped.")
            return None

    def generate_and_save(
        self, config: GenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """Generate audio and save to file. Returns the output filepath."""
        errors = config.validate()
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))

        if not config.output_filename:
            mode_name = config.mode if config.mode != "custom" else f"custom_{config.beat_frequency}hz"
            config.output_filename = f"brainwave_{mode_name}_{config.tone_type}_{int(config.duration_seconds)}s.wav"

        filepath = os.path.join(config.output_directory, config.output_filename)
        audio_data, sr = self.generate(config, progress_callback)
        self.save_wav(audio_data, sr, filepath)

        if config.export_flac:
            flac_path = filepath.rsplit(".", 1)[0] + ".flac"
            self.save_flac(audio_data, sr, flac_path)

        if progress_callback:
            progress_callback(100, f"Done! Saved to {filepath}")

        return filepath

    @staticmethod
    def estimate_file_size(config: GenerationConfig) -> int:
        """Estimate WAV file size in bytes."""
        num_samples = int(config.sample_rate * config.duration_seconds)
        return 44 + (num_samples * 2 * 2)  # header + samples × channels × bytes

    @staticmethod
    def get_preview_data(config: GenerationConfig, preview_seconds: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Generate short preview for waveform display. Returns (time, audio_mono)."""
        engine = AudioEngine()
        sr = config.sample_rate
        dur = min(preview_seconds, config.duration_seconds)
        beat_freq = config.beat_frequency
        if config.mode != "custom" and config.mode in BRAINWAVE_MODES:
            beat_freq = BRAINWAVE_MODES[config.mode].beat_frequency
        carrier = config.carrier_frequency
        tone_type = ToneType(config.tone_type)

        if tone_type == ToneType.PURE:
            audio = engine.generate_sine(carrier, dur, sr)
        elif tone_type == ToneType.BINAURAL:
            left, right = engine.generate_binaural(carrier, beat_freq, dur, sr)
            audio = (left + right) / 2.0
        elif tone_type == ToneType.MONAURAL:
            audio = engine.generate_monaural(carrier, beat_freq, dur, sr)
        elif tone_type == ToneType.ISOCHRONIC:
            audio = engine.generate_isochronic(carrier, beat_freq, dur, sr)
        else:
            audio = engine.generate_sine(carrier, dur, sr)

        audio = engine.normalize(audio) * config.volume
        t = np.linspace(0, dur, len(audio), endpoint=False)
        return t, audio


# ============================================================
# CONFIGURATION PERSISTENCE
# ============================================================

class ConfigManager:
    """Save and load generation configurations as JSON."""
    DEFAULT_PATH = "brainwave_config.json"

    @staticmethod
    def save(config: GenerationConfig, filepath: str = None) -> str:
        filepath = filepath or ConfigManager.DEFAULT_PATH
        with open(filepath, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str = None) -> GenerationConfig:
        filepath = filepath or ConfigManager.DEFAULT_PATH
        with open(filepath, "r") as f:
            data = json.load(f)
        config = GenerationConfig.from_dict(data)
        logger.info(f"Configuration loaded from {filepath}")
        return config


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CLI mode."""
    parser = argparse.ArgumentParser(
        description="🧠 Brainwave Audio Tone Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python solution.py cli --mode relax --duration 300
  python solution.py cli --mode custom --beat-freq 12 --tone-type binaural
  python solution.py cli --mode sleep --noise white --noise-intensity 0.4
  python solution.py cli --generate-all
  python solution.py gui
        """,
    )
    subparsers = parser.add_subparsers(dest="interface", help="Interface mode")

    # CLI subcommand
    cli_parser = subparsers.add_parser("cli", help="Command-line interface")
    mode_choices = list(BRAINWAVE_MODES.keys()) + ["custom"]
    cli_parser.add_argument("--mode", "-m", choices=mode_choices, default="relax")
    cli_parser.add_argument("--duration", "-d", type=float, default=300, help="Duration in seconds")
    cli_parser.add_argument("--duration-minutes", type=float, default=None)
    cli_parser.add_argument("--volume", "-v", type=float, default=0.7)
    cli_parser.add_argument("--carrier", "-c", type=float, default=200.0)
    cli_parser.add_argument("--beat-freq", type=float, default=None)
    cli_parser.add_argument("--tone-type", "-t", choices=[t.value for t in ToneType], default="binaural")
    cli_parser.add_argument("--sample-rate", type=int, choices=[22050, 44100, 48000, 96000], default=44100)
    cli_parser.add_argument("--noise", choices=[n.value for n in NoiseType], default="none")
    cli_parser.add_argument("--noise-intensity", type=float, default=0.3)
    cli_parser.add_argument("--fade", type=float, default=2.0)
    cli_parser.add_argument("--output", "-o", type=str, default="")
    cli_parser.add_argument("--output-dir", type=str, default="generated_audio")
    cli_parser.add_argument("--flac", action="store_true")
    cli_parser.add_argument("--generate-all", action="store_true")
    cli_parser.add_argument("--save-config", type=str, default=None)
    cli_parser.add_argument("--load-config", type=str, default=None)

    # GUI subcommand
    subparsers.add_parser("gui", help="Graphical user interface")
    return parser


def run_cli(args) -> None:
    """Execute CLI mode."""
    generator = BrainwaveGenerator()

    if args.load_config:
        config = ConfigManager.load(args.load_config)
    else:
        duration = args.duration
        if args.duration_minutes is not None:
            duration = args.duration_minutes * 60.0
        beat_freq = args.beat_freq if args.beat_freq is not None else 8.0
        if args.mode != "custom" and args.mode in BRAINWAVE_MODES:
            beat_freq = BRAINWAVE_MODES[args.mode].beat_frequency
        config = GenerationConfig(
            mode=args.mode, beat_frequency=beat_freq,
            carrier_frequency=args.carrier, tone_type=args.tone_type,
            duration_seconds=duration, volume=args.volume,
            sample_rate=args.sample_rate, noise_type=args.noise,
            noise_intensity=args.noise_intensity, fade_duration=args.fade,
            output_filename=args.output, output_directory=args.output_dir,
            export_flac=args.flac,
        )

    if args.save_config:
        ConfigManager.save(config, args.save_config)

    if args.generate_all:
        logger.info("=" * 60)
        logger.info("BATCH GENERATION: All predefined modes")
        logger.info("=" * 60)
        for mode_name, mode_def in BRAINWAVE_MODES.items():
            batch_config = GenerationConfig(
                mode=mode_name, beat_frequency=mode_def.beat_frequency,
                carrier_frequency=config.carrier_frequency, tone_type=config.tone_type,
                duration_seconds=config.duration_seconds, volume=config.volume,
                sample_rate=config.sample_rate, noise_type=config.noise_type,
                noise_intensity=config.noise_intensity, fade_duration=config.fade_duration,
                output_directory=config.output_directory, export_flac=config.export_flac,
            )
            try:
                filepath = generator.generate_and_save(batch_config)
                logger.info(f"✓ {mode_name}: {filepath}")
            except Exception as e:
                logger.error(f"✗ {mode_name}: {e}")
        logger.info("Batch generation complete.")
        return

    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Validation error: {err}")
        sys.exit(1)

    estimated_size = BrainwaveGenerator.estimate_file_size(config)
    num_samples = int(config.sample_rate * config.duration_seconds)

    logger.info("=" * 60)
    logger.info("🧠 BRAINWAVE AUDIO GENERATOR")
    logger.info("=" * 60)
    if config.mode in BRAINWAVE_MODES:
        mi = BRAINWAVE_MODES[config.mode]
        logger.info(f"Mode: {mi.emoji} {mi.description}")
        logger.info(f"Band: {mi.band.upper()}")
    else:
        logger.info(f"Mode: Custom ({config.beat_frequency} Hz)")
    logger.info(f"Tone Type: {config.tone_type}")
    logger.info(f"Carrier: {config.carrier_frequency} Hz")
    logger.info(f"Beat: {config.beat_frequency} Hz")
    logger.info(f"Duration: {config.duration_seconds:.1f}s ({config.duration_seconds / 60:.1f} min)")
    logger.info(f"Volume: {config.volume}")
    logger.info(f"Sample Rate: {config.sample_rate} Hz")
    logger.info(f"Samples: {num_samples:,}")
    logger.info(f"Est. Size: {estimated_size / 1024 / 1024:.2f} MB")
    logger.info(f"Noise: {config.noise_type} ({config.noise_intensity * 100:.0f}%)")
    logger.info(f"Fade: {config.fade_duration}s")
    logger.info("-" * 60)

    filepath = generator.generate_and_save(config)
    logger.info("=" * 60)
    logger.info(f"Output: {os.path.abspath(filepath)}")
    logger.info("✅ Generation complete!")


# ============================================================
# GRAPHICAL USER INTERFACE (Tkinter) — COLORFUL & RESPONSIVE
# ============================================================

def run_gui() -> None:
    """Launch the beautiful Tkinter GUI."""
    import tkinter as tk
    from tkinter import messagebox, filedialog

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    # ======== COLOR THEMES ========
    THEMES = {
        "light": {
            "name": "Light",
            "bg": "#F5F7FA",
            "bg2": "#FFFFFF",
            "bg3": "#EEF1F5",
            "fg": "#2D3436",
            "fg2": "#636E72",
            "accent": "#6C5CE7",
            "accent2": "#A29BFE",
            "success": "#00B894",
            "warning": "#FDCB6E",
            "error": "#E17055",
            "card_bg": "#FFFFFF",
            "card_border": "#DFE6E9",
            "entry_bg": "#FFFFFF",
            "entry_fg": "#2D3436",
            "entry_border": "#B2BEC3",
            "btn_primary": "#6C5CE7",
            "btn_primary_fg": "#FFFFFF",
            "btn_secondary": "#DFE6E9",
            "btn_secondary_fg": "#2D3436",
            "btn_success": "#00B894",
            "btn_success_fg": "#FFFFFF",
            "btn_warning": "#FDCB6E",
            "btn_warning_fg": "#2D3436",
            "slider_trough": "#DFE6E9",
            "slider_fg": "#6C5CE7",
            "header_bg": "#6C5CE7",
            "header_fg": "#FFFFFF",
            "plot_bg": "#FFFFFF",
            "plot_fg": "#2D3436",
            "plot_line": "#6C5CE7",
            "plot_line2": "#E17055",
            "progress_bg": "#DFE6E9",
            "progress_fg": "#6C5CE7",
            "divider": "#DFE6E9",
            "shadow": "#B2BEC3",
            "tag_bg": "#EEF1F5",
            "scrollbar": "#B2BEC3",
        },
        "dark": {
            "name": "Dark",
            "bg": "#1A1A2E",
            "bg2": "#16213E",
            "bg3": "#0F3460",
            "fg": "#E8E8E8",
            "fg2": "#A0A0B0",
            "accent": "#E94560",
            "accent2": "#FF6B6B",
            "success": "#00D2D3",
            "warning": "#FECA57",
            "error": "#FF6B6B",
            "card_bg": "#16213E",
            "card_border": "#0F3460",
            "entry_bg": "#0F3460",
            "entry_fg": "#E8E8E8",
            "entry_border": "#533483",
            "btn_primary": "#E94560",
            "btn_primary_fg": "#FFFFFF",
            "btn_secondary": "#0F3460",
            "btn_secondary_fg": "#E8E8E8",
            "btn_success": "#00D2D3",
            "btn_success_fg": "#1A1A2E",
            "btn_warning": "#FECA57",
            "btn_warning_fg": "#1A1A2E",
            "slider_trough": "#0F3460",
            "slider_fg": "#E94560",
            "header_bg": "#E94560",
            "header_fg": "#FFFFFF",
            "plot_bg": "#16213E",
            "plot_fg": "#E8E8E8",
            "plot_line": "#E94560",
            "plot_line2": "#FECA57",
            "progress_bg": "#0F3460",
            "progress_fg": "#E94560",
            "divider": "#0F3460",
            "shadow": "#0A0A1A",
            "tag_bg": "#0F3460",
            "scrollbar": "#533483",
        },
    }

    class RoundedButton(tk.Canvas):
        """A custom rounded button with hover effects and colors."""
        def __init__(self, parent, text="", command=None, width=140, height=38,
                     bg_color="#6C5CE7", fg_color="#FFFFFF", hover_color=None,
                     font_size=10, corner_radius=10, emoji="", **kwargs):
            super().__init__(parent, width=width, height=height,
                             highlightthickness=0, borderwidth=0, **kwargs)

            self._command = command
            self._bg = bg_color
            self._fg = fg_color
            self._hover = hover_color or self._lighten(bg_color, 30)
            self._text = text
            self._emoji = emoji
            self._width = width
            self._height = height
            self._radius = corner_radius
            self._font_size = font_size
            self._pressed = False
            self._enabled = True

            self._draw(self._bg)

            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<ButtonPress-1>", self._on_press)
            self.bind("<ButtonRelease-1>", self._on_release)

        def _lighten(self, color, amount):
            """Lighten a hex color."""
            color = color.lstrip("#")
            r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            r = min(255, r + amount)
            g = min(255, g + amount)
            b = min(255, b + amount)
            return f"#{r:02x}{g:02x}{b:02x}"

        def _darken(self, color, amount):
            """Darken a hex color."""
            color = color.lstrip("#")
            r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            r = max(0, r - amount)
            g = max(0, g - amount)
            b = max(0, b - amount)
            return f"#{r:02x}{g:02x}{b:02x}"

        def _draw(self, fill_color):
            self.delete("all")
            r = self._radius
            w, h = self._width, self._height
            # Rounded rectangle
            self.create_arc(0, 0, 2 * r, 2 * r, start=90, extent=90, fill=fill_color, outline="")
            self.create_arc(w - 2 * r, 0, w, 2 * r, start=0, extent=90, fill=fill_color, outline="")
            self.create_arc(0, h - 2 * r, 2 * r, h, start=180, extent=90, fill=fill_color, outline="")
            self.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=fill_color, outline="")
            self.create_rectangle(r, 0, w - r, h, fill=fill_color, outline="")
            self.create_rectangle(0, r, w, h - r, fill=fill_color, outline="")
            # Text
            display_text = f"{self._emoji} {self._text}" if self._emoji else self._text
            self.create_text(w / 2, h / 2, text=display_text, fill=self._fg,
                             font=("Segoe UI", self._font_size, "bold"))

        def _on_enter(self, e):
            if self._enabled:
                self._draw(self._hover)
                self.configure(cursor="hand2")

        def _on_leave(self, e):
            if self._enabled:
                self._draw(self._bg)
                self.configure(cursor="")

        def _on_press(self, e):
            if self._enabled:
                self._draw(self._darken(self._bg, 30))

        def _on_release(self, e):
            if self._enabled:
                self._draw(self._hover)
                if self._command:
                    self._command()

        def set_enabled(self, enabled: bool):
            self._enabled = enabled
            if not enabled:
                self._draw(self._lighten(self._bg, 60))
            else:
                self._draw(self._bg)

        def update_colors(self, bg, fg, hover=None):
            self._bg = bg
            self._fg = fg
            self._hover = hover or self._lighten(bg, 30)
            self._draw(self._bg)

    class StyledEntry(tk.Frame):
        """Custom styled entry with label, rounded border and focus effects."""
        def __init__(self, parent, label="", variable=None, width=120,
                     bg="#FFFFFF", fg="#2D3436", border_color="#B2BEC3",
                     accent="#6C5CE7", label_fg="#636E72", **kwargs):
            super().__init__(parent, bg=kwargs.get("master_bg", bg))
            self._accent = accent
            self._border_color = border_color
            self._bg = bg
            self._fg = fg
            self._label_fg = label_fg

            if label:
                self._label = tk.Label(self, text=label, font=("Segoe UI", 9),
                                       fg=label_fg, bg=self.cget("bg"))
                self._label.pack(anchor="w", padx=2)

            self._border_frame = tk.Frame(self, bg=border_color, padx=2, pady=2)
            self._border_frame.pack(fill="x")

            self.entry = tk.Entry(
                self._border_frame, textvariable=variable,
                font=("Segoe UI", 11), bg=bg, fg=fg,
                relief="flat", width=width // 10,
                insertbackground=fg, selectbackground=accent,
                selectforeground="#FFFFFF",
            )
            self.entry.pack(fill="x", padx=1, pady=1)

            self.entry.bind("<FocusIn>", lambda e: self._border_frame.configure(bg=accent))
            self.entry.bind("<FocusOut>", lambda e: self._border_frame.configure(bg=border_color))

        def update_colors(self, bg, fg, border, accent, label_fg, master_bg):
            self.configure(bg=master_bg)
            self._bg = bg
            self._fg = fg
            self._border_color = border
            self._accent = accent
            self._border_frame.configure(bg=border)
            self.entry.configure(bg=bg, fg=fg, insertbackground=fg, selectbackground=accent)
            if hasattr(self, "_label"):
                self._label.configure(fg=label_fg, bg=master_bg)

    class ModeCard(tk.Frame):
        """A clickable card for selecting a brainwave mode."""
        def __init__(self, parent, mode_name, mode_def, selected_var,
                     on_select, theme, **kwargs):
            super().__init__(parent, **kwargs)
            self.mode_name = mode_name
            self.mode_def = mode_def
            self.selected_var = selected_var
            self.on_select = on_select
            self._selected = False

            self.configure(
                bg=theme["card_bg"], relief="flat",
                highlightthickness=2, highlightbackground=theme["card_border"],
                padx=8, pady=6, cursor="hand2",
            )

            emoji_label = tk.Label(self, text=mode_def.emoji, font=("Segoe UI Emoji", 18),
                                   bg=theme["card_bg"])
            emoji_label.pack(side="left", padx=(4, 8))

            text_frame = tk.Frame(self, bg=theme["card_bg"])
            text_frame.pack(side="left", fill="x", expand=True)

            name_label = tk.Label(text_frame, text=mode_name.replace("_", " ").title(),
                                  font=("Segoe UI", 10, "bold"),
                                  fg=theme["fg"], bg=theme["card_bg"])
            name_label.pack(anchor="w")

            desc_label = tk.Label(text_frame, text=f"{mode_def.beat_frequency} Hz · {mode_def.band.upper()}",
                                  font=("Segoe UI", 8),
                                  fg=theme["fg2"], bg=theme["card_bg"])
            desc_label.pack(anchor="w")

            # Band color indicator
            band_color = mode_def.color
            indicator = tk.Frame(self, bg=band_color, width=4)
            indicator.pack(side="right", fill="y", padx=(8, 0))

            # Bind click to all children
            for widget in [self, emoji_label, text_frame, name_label, desc_label, indicator]:
                widget.bind("<Button-1>", self._on_click)
                widget.configure(cursor="hand2")

            self._children_widgets = [self, emoji_label, text_frame, name_label, desc_label]
            self._indicator = indicator
            self._name_label = name_label
            self._desc_label = desc_label
            self._emoji_label = emoji_label
            self._text_frame = text_frame

        def _on_click(self, e=None):
            self.selected_var.set(self.mode_name)
            self.on_select()

        def set_selected(self, selected: bool, theme: dict):
            self._selected = selected
            if selected:
                self.configure(highlightbackground=self.mode_def.color, highlightthickness=3)
                bg = theme["accent"] + "15" if len(theme["accent"]) == 7 else theme["bg3"]
                # Use the bg3 for selection highlight
                for w in [self, self._emoji_label, self._text_frame,
                           self._name_label, self._desc_label]:
                    w.configure(bg=theme["bg3"])
            else:
                self.configure(highlightbackground=theme["card_border"], highlightthickness=2)
                for w in [self, self._emoji_label, self._text_frame,
                           self._name_label, self._desc_label]:
                    w.configure(bg=theme["card_bg"])

        def update_theme(self, theme):
            selected = self._selected
            base_bg = theme["bg3"] if selected else theme["card_bg"]
            self.configure(bg=base_bg)
            if selected:
                self.configure(highlightbackground=self.mode_def.color, highlightthickness=3)
            else:
                self.configure(highlightbackground=theme["card_border"], highlightthickness=2)
            for w in [self._emoji_label, self._text_frame]:
                w.configure(bg=base_bg)
            self._name_label.configure(bg=base_bg, fg=theme["fg"])
            self._desc_label.configure(bg=base_bg, fg=theme["fg2"])

    class BrainwaveApp:
        """Main GUI Application — Colorful, Responsive, Easy to Use."""

        def __init__(self):
            self.root = tk.Tk()
            self.root.title("🧠 Brainwave Audio Generator")
            self.root.geometry("1200x900")
            self.root.minsize(1000, 750)
            self.root.configure(bg="#F5F7FA")

            # Try to set icon (won't fail if unavailable)
            try:
                self.root.iconbitmap(default="")
            except Exception:
                pass

            self.generator = BrainwaveGenerator()
            self.current_theme = "light"
            self.generating = False
            self.mode_cards: List[ModeCard] = []

            # ---- Tk Variables ----
            self.mode_var = tk.StringVar(value="relax")
            self.tone_type_var = tk.StringVar(value="binaural")
            self.duration_min_var = tk.StringVar(value="5")
            self.duration_sec_var = tk.StringVar(value="0")
            self.carrier_var = tk.StringVar(value="200")
            self.volume_var = tk.DoubleVar(value=0.7)
            self.beat_freq_var = tk.StringVar(value="8.0")
            self.sample_rate_var = tk.StringVar(value="44100")
            self.noise_type_var = tk.StringVar(value="none")
            self.noise_intensity_var = tk.DoubleVar(value=0.3)
            self.fade_var = tk.StringVar(value="2.0")
            self.output_dir_var = tk.StringVar(value="generated_audio")
            self.output_file_var = tk.StringVar(value="")
            self.export_flac_var = tk.BooleanVar(value=False)
            self.progress_var = tk.DoubleVar(value=0.0)
            self.status_var = tk.StringVar(value="✨ Ready to generate brainwave audio!")

            self._build_ui()
            self._apply_theme()
            self._on_mode_change()

            # Make responsive
            self.root.bind("<Configure>", self._on_resize)

        def _theme(self):
            return THEMES[self.current_theme]

        def _build_ui(self):
            """Build the complete gorgeous UI."""
            theme = self._theme()
            root = self.root

            # ========== HEADER BAR ==========
            self.header_frame = tk.Frame(root, bg=theme["header_bg"], height=60)
            self.header_frame.pack(fill="x")
            self.header_frame.pack_propagate(False)

            self.header_title = tk.Label(
                self.header_frame, text="🧠  Brainwave Audio Generator",
                font=("Segoe UI", 18, "bold"), fg=theme["header_fg"],
                bg=theme["header_bg"]
            )
            self.header_title.pack(side="left", padx=20, pady=10)

            self.header_subtitle = tk.Label(
                self.header_frame, text="Generate • Relax • Focus • Sleep",
                font=("Segoe UI", 10), fg=theme["header_fg"],
                bg=theme["header_bg"]
            )
            self.header_subtitle.pack(side="left", padx=10, pady=10)

            # Theme toggle button
            self.theme_btn = RoundedButton(
                self.header_frame, text="Dark Mode", emoji="🌙",
                command=self._toggle_theme, width=140, height=34,
                bg_color="#FFFFFF", fg_color=theme["header_bg"],
                font_size=9
            )
            self.theme_btn.pack(side="right", padx=20, pady=13)
            self.theme_btn.configure(bg=theme["header_bg"])

            # ========== MAIN CONTENT ==========
            self.main_frame = tk.Frame(root, bg=theme["bg"])
            self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)

            # Left panel (scrollable)
            self.left_canvas = tk.Canvas(self.main_frame, bg=theme["bg"],
                                         highlightthickness=0, borderwidth=0)
            self.left_scrollbar = tk.Scrollbar(self.main_frame, orient="vertical",
                                                command=self.left_canvas.yview)
            self.left_scroll_frame = tk.Frame(self.left_canvas, bg=theme["bg"])

            self.left_scroll_frame.bind(
                "<Configure>",
                lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
            )

            self.left_canvas_window = self.left_canvas.create_window(
                (0, 0), window=self.left_scroll_frame, anchor="nw"
            )
            self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

            self.left_canvas.pack(side="left", fill="both", expand=True)
            self.left_scrollbar.pack(side="left", fill="y")

            # Bind mousewheel
            def _on_mousewheel(event):
                self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

            # Right panel (preview)
            self.right_frame = tk.Frame(self.main_frame, bg=theme["bg"], width=380)
            self.right_frame.pack(side="right", fill="both", padx=0)
            self.right_frame.pack_propagate(False)

            # Build left panel content
            self._build_left_panel(self.left_scroll_frame, theme)

            # Build right panel content
            self._build_right_panel(self.right_frame, theme)

            # ========== BOTTOM BAR ==========
            self._build_bottom_bar(root, theme)

            # Update canvas width on resize
            self.left_canvas.bind("<Configure>", self._on_canvas_configure)

        def _on_canvas_configure(self, event):
            self.left_canvas.itemconfig(self.left_canvas_window, width=event.width)

        def _on_resize(self, event=None):
            """Handle window resize for responsiveness."""
            pass

        def _build_left_panel(self, parent, theme):
            """Build the left settings panel."""
            pad = {"padx": 15, "pady": 6}

            # ---- Section: Mode Selection (Card Grid) ----
            self._section_label(parent, "🎵  Select Brainwave Mode", theme)

            self.modes_frame = tk.Frame(parent, bg=theme["bg"])
            self.modes_frame.pack(fill="x", **pad)

            self.mode_cards = []
            row_frame = None
            for i, (mode_name, mode_def) in enumerate(BRAINWAVE_MODES.items()):
                if i % 2 == 0:
                    row_frame = tk.Frame(self.modes_frame, bg=theme["bg"])
                    row_frame.pack(fill="x", pady=3)

                card = ModeCard(
                    row_frame, mode_name, mode_def,
                    self.mode_var, self._on_mode_change, theme
                )
                card.pack(side="left", fill="x", expand=True, padx=4, pady=2)
                self.mode_cards.append(card)

            # Custom mode option
            custom_frame = tk.Frame(self.modes_frame, bg=theme["bg"])
            custom_frame.pack(fill="x", pady=3)

            self.custom_card = tk.Frame(
                custom_frame, bg=theme["card_bg"], relief="flat",
                highlightthickness=2, highlightbackground=theme["card_border"],
                padx=8, pady=6, cursor="hand2"
            )
            self.custom_card.pack(fill="x", padx=4, pady=2)

            custom_inner = tk.Frame(self.custom_card, bg=theme["card_bg"])
            custom_inner.pack(fill="x")

            self.custom_emoji = tk.Label(custom_inner, text="⚙️", font=("Segoe UI Emoji", 16),
                                         bg=theme["card_bg"])
            self.custom_emoji.pack(side="left", padx=(4, 8))

            self.custom_label = tk.Label(custom_inner, text="Custom Mode — Set your own frequency",
                                          font=("Segoe UI", 10, "bold"),
                                          fg=theme["fg"], bg=theme["card_bg"])
            self.custom_label.pack(side="left")

            for w in [self.custom_card, custom_inner, self.custom_emoji, self.custom_label]:
                w.bind("<Button-1>", lambda e: self._select_custom())
                w.configure(cursor="hand2")

            # ---- Section: Tone Type ----
            self._section_label(parent, "🔊  Tone Type", theme)

            self.tone_frame = tk.Frame(parent, bg=theme["bg"])
            self.tone_frame.pack(fill="x", **pad)

            self.tone_buttons = {}
            tone_info = {
                "binaural": ("🎧", "Binaural", "Different freq each ear"),
                "monaural": ("📢", "Monaural", "AM modulated single tone"),
                "isochronic": ("⚡", "Isochronic", "Sharp on/off pulses"),
                "pure": ("〰️", "Pure Sine", "Single frequency tone"),
            }

            for tt_value, (emoji, name, desc) in tone_info.items():
                btn_frame = tk.Frame(self.tone_frame, bg=theme["card_bg"],
                                      relief="flat", highlightthickness=2,
                                      highlightbackground=theme["card_border"],
                                      padx=8, pady=8, cursor="hand2")
                btn_frame.pack(side="left", fill="x", expand=True, padx=3)

                e_lbl = tk.Label(btn_frame, text=emoji, font=("Segoe UI Emoji", 14),
                                 bg=theme["card_bg"])
                e_lbl.pack()
                n_lbl = tk.Label(btn_frame, text=name, font=("Segoe UI", 9, "bold"),
                                 fg=theme["fg"], bg=theme["card_bg"])
                n_lbl.pack()
                d_lbl = tk.Label(btn_frame, text=desc, font=("Segoe UI", 7),
                                 fg=theme["fg2"], bg=theme["card_bg"])
                d_lbl.pack()

                self.tone_buttons[tt_value] = {
                    "frame": btn_frame, "emoji": e_lbl,
                    "name": n_lbl, "desc": d_lbl
                }

                for w in [btn_frame, e_lbl, n_lbl, d_lbl]:
                    w.bind("<Button-1>", lambda e, v=tt_value: self._select_tone(v))
                    w.configure(cursor="hand2")

            # ---- Section: Frequency & Duration ----
            self._section_label(parent, "⚙️  Settings", theme)

            settings_frame = tk.Frame(parent, bg=theme["card_bg"],
                                       relief="flat", highlightthickness=1,
                                       highlightbackground=theme["card_border"])
            settings_frame.pack(fill="x", **pad)

            # Row 1: Carrier + Beat + Sample Rate
            row1 = tk.Frame(settings_frame, bg=theme["card_bg"])
            row1.pack(fill="x", padx=12, pady=(10, 4))

            self.carrier_entry = StyledEntry(
                row1, label="Carrier Freq (Hz)", variable=self.carrier_var,
                width=120, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.carrier_entry.pack(side="left", padx=6, fill="x", expand=True)

            self.beat_entry = StyledEntry(
                row1, label="Beat Freq (Hz)", variable=self.beat_freq_var,
                width=120, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.beat_entry.pack(side="left", padx=6, fill="x", expand=True)

            sr_frame = tk.Frame(row1, bg=theme["card_bg"])
            sr_frame.pack(side="left", padx=6, fill="x", expand=True)
            tk.Label(sr_frame, text="Sample Rate", font=("Segoe UI", 9),
                     fg=theme["fg2"], bg=theme["card_bg"]).pack(anchor="w", padx=2)
            self.sr_menu = tk.OptionMenu(sr_frame, self.sample_rate_var,
                                          "22050", "44100", "48000", "96000")
            self.sr_menu.configure(font=("Segoe UI", 10), width=8,
                                    bg=theme["entry_bg"], fg=theme["entry_fg"],
                                    relief="flat", highlightthickness=1,
                                    highlightbackground=theme["entry_border"],
                                    activebackground=theme["accent"],
                                    activeforeground="#FFFFFF")
            self.sr_menu.pack(fill="x", padx=1)

            # Row 2: Duration + Volume + Fade
            row2 = tk.Frame(settings_frame, bg=theme["card_bg"])
            row2.pack(fill="x", padx=12, pady=4)

            self.dur_min_entry = StyledEntry(
                row2, label="Minutes", variable=self.duration_min_var,
                width=80, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.dur_min_entry.pack(side="left", padx=6, fill="x", expand=True)

            self.dur_sec_entry = StyledEntry(
                row2, label="Seconds", variable=self.duration_sec_var,
                width=80, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.dur_sec_entry.pack(side="left", padx=6, fill="x", expand=True)

            # Volume slider
            vol_frame = tk.Frame(row2, bg=theme["card_bg"])
            vol_frame.pack(side="left", padx=6, fill="x", expand=True)

            vol_label_frame = tk.Frame(vol_frame, bg=theme["card_bg"])
            vol_label_frame.pack(fill="x")
            tk.Label(vol_label_frame, text="Volume", font=("Segoe UI", 9),
                     fg=theme["fg2"], bg=theme["card_bg"]).pack(side="left", padx=2)
            self.vol_display = tk.Label(vol_label_frame, text="70%",
                                         font=("Segoe UI", 9, "bold"),
                                         fg=theme["accent"], bg=theme["card_bg"])
            self.vol_display.pack(side="right", padx=2)

            self.volume_slider = tk.Scale(
                vol_frame, from_=0.0, to=1.0, resolution=0.01,
                orient="horizontal", variable=self.volume_var,
                showvalue=False, bg=theme["card_bg"],
                troughcolor=theme["slider_trough"],
                fg=theme["accent"], highlightthickness=0,
                sliderrelief="flat", activebackground=theme["accent"],
                length=150,
                command=lambda v: self.vol_display.configure(
                    text=f"{float(v) * 100:.0f}%")
            )
            self.volume_slider.pack(fill="x")

            # Fade
            self.fade_entry = StyledEntry(
                row2, label="Fade (sec)", variable=self.fade_var,
                width=80, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.fade_entry.pack(side="left", padx=6, fill="x", expand=True)

            # ---- Section: Background Noise ----
            self._section_label(parent, "🌿  Background Noise", theme)

            noise_card = tk.Frame(parent, bg=theme["card_bg"],
                                   relief="flat", highlightthickness=1,
                                   highlightbackground=theme["card_border"])
            noise_card.pack(fill="x", **pad)
            self.noise_card = noise_card

            noise_row = tk.Frame(noise_card, bg=theme["card_bg"])
            noise_row.pack(fill="x", padx=12, pady=10)

            # Noise type buttons
            self.noise_buttons = {}
            noise_info = {
                "none": ("🔇", "None"),
                "white": ("⬜", "White"),
                "pink": ("🩷", "Pink"),
                "brown": ("🟫", "Brown"),
            }

            for nt_value, (emoji, name) in noise_info.items():
                nb_frame = tk.Frame(noise_row, bg=theme["card_bg"],
                                     relief="flat", highlightthickness=2,
                                     highlightbackground=theme["card_border"],
                                     padx=12, pady=6, cursor="hand2")
                nb_frame.pack(side="left", padx=4)

                nb_label = tk.Label(nb_frame, text=f"{emoji} {name}",
                                     font=("Segoe UI", 10),
                                     fg=theme["fg"], bg=theme["card_bg"],
                                     cursor="hand2")
                nb_label.pack()

                self.noise_buttons[nt_value] = {"frame": nb_frame, "label": nb_label}

                for w in [nb_frame, nb_label]:
                    w.bind("<Button-1>", lambda e, v=nt_value: self._select_noise(v))

            # Noise intensity slider
            ni_frame = tk.Frame(noise_card, bg=theme["card_bg"])
            ni_frame.pack(fill="x", padx=12, pady=(0, 10))

            ni_label_frame = tk.Frame(ni_frame, bg=theme["card_bg"])
            ni_label_frame.pack(fill="x")
            tk.Label(ni_label_frame, text="Noise Intensity",
                     font=("Segoe UI", 9), fg=theme["fg2"],
                     bg=theme["card_bg"]).pack(side="left")
            self.noise_pct_label = tk.Label(ni_label_frame, text="30%",
                                             font=("Segoe UI", 9, "bold"),
                                             fg=theme["accent"],
                                             bg=theme["card_bg"])
            self.noise_pct_label.pack(side="right")

            self.noise_slider = tk.Scale(
                ni_frame, from_=0.0, to=1.0, resolution=0.01,
                orient="horizontal", variable=self.noise_intensity_var,
                showvalue=False, bg=theme["card_bg"],
                troughcolor=theme["slider_trough"],
                fg=theme["accent"], highlightthickness=0,
                sliderrelief="flat", activebackground=theme["accent"],
                command=lambda v: self.noise_pct_label.configure(
                    text=f"{float(v) * 100:.0f}%")
            )
            self.noise_slider.pack(fill="x")

            # ---- Section: Output ----
            self._section_label(parent, "💾  Output Settings", theme)

            out_card = tk.Frame(parent, bg=theme["card_bg"],
                                 relief="flat", highlightthickness=1,
                                 highlightbackground=theme["card_border"])
            out_card.pack(fill="x", **pad)
            self.out_card = out_card

            out_row1 = tk.Frame(out_card, bg=theme["card_bg"])
            out_row1.pack(fill="x", padx=12, pady=(10, 4))

            self.outdir_entry = StyledEntry(
                out_row1, label="Output Directory", variable=self.output_dir_var,
                width=250, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.outdir_entry.pack(side="left", padx=6, fill="x", expand=True)

            self.browse_btn = RoundedButton(
                out_row1, text="Browse", emoji="📁",
                command=self._browse_dir, width=110, height=34,
                bg_color=theme["btn_secondary"],
                fg_color=theme["btn_secondary_fg"],
                font_size=9
            )
            self.browse_btn.pack(side="right", padx=6, pady=(16, 0))
            self.browse_btn.configure(bg=theme["card_bg"])

            out_row2 = tk.Frame(out_card, bg=theme["card_bg"])
            out_row2.pack(fill="x", padx=12, pady=(0, 10))

            self.outfile_entry = StyledEntry(
                out_row2, label="Filename (auto if empty)", variable=self.output_file_var,
                width=250, bg=theme["entry_bg"], fg=theme["entry_fg"],
                border_color=theme["entry_border"], accent=theme["accent"],
                label_fg=theme["fg2"], master_bg=theme["card_bg"]
            )
            self.outfile_entry.pack(side="left", padx=6, fill="x", expand=True)

            self.flac_var_frame = tk.Frame(out_row2, bg=theme["card_bg"])
            self.flac_var_frame.pack(side="right", padx=6)
            self.flac_cb = tk.Checkbutton(
                self.flac_var_frame, text="Export FLAC",
                variable=self.export_flac_var,
                font=("Segoe UI", 9), fg=theme["fg"],
                bg=theme["card_bg"], selectcolor=theme["entry_bg"],
                activebackground=theme["card_bg"],
                activeforeground=theme["fg"]
            )
            self.flac_cb.pack(pady=(16, 0))

            # ---- Action Buttons ----
            self._section_label(parent, "", theme)

            btn_row = tk.Frame(parent, bg=theme["bg"])
            btn_row.pack(fill="x", padx=15, pady=8)

            self.gen_btn = RoundedButton(
                btn_row, text="Generate", emoji="▶",
                command=self._on_generate, width=180, height=46,
                bg_color=theme["btn_primary"],
                fg_color=theme["btn_primary_fg"],
                font_size=12, corner_radius=12
            )
            self.gen_btn.pack(side="left", padx=6)
            self.gen_btn.configure(bg=theme["bg"])

            self.gen_all_btn = RoundedButton(
                btn_row, text="Generate All", emoji="🔄",
                command=self._on_generate_all, width=180, height=46,
                bg_color=theme["btn_success"],
                fg_color=theme["btn_success_fg"],
                font_size=12, corner_radius=12
            )
            self.gen_all_btn.pack(side="left", padx=6)
            self.gen_all_btn.configure(bg=theme["bg"])

            self.preview_btn = RoundedButton(
                btn_row, text="Preview", emoji="👁",
                command=self._update_preview, width=130, height=46,
                bg_color=theme["btn_warning"],
                fg_color=theme["btn_warning_fg"],
                font_size=11, corner_radius=12
            )
            self.preview_btn.pack(side="left", padx=6)
            self.preview_btn.configure(bg=theme["bg"])

            self.save_cfg_btn = RoundedButton(
                btn_row, text="Save Config", emoji="💾",
                command=self._save_config, width=140, height=46,
                bg_color=theme["btn_secondary"],
                fg_color=theme["btn_secondary_fg"],
                font_size=10, corner_radius=12
            )
            self.save_cfg_btn.pack(side="left", padx=6)
            self.save_cfg_btn.configure(bg=theme["bg"])

            self.load_cfg_btn = RoundedButton(
                btn_row, text="Load Config", emoji="📂",
                command=self._load_config, width=140, height=46,
                bg_color=theme["btn_secondary"],
                fg_color=theme["btn_secondary_fg"],
                font_size=10, corner_radius=12
            )
            self.load_cfg_btn.pack(side="left", padx=6)
            self.load_cfg_btn.configure(bg=theme["bg"])

        def _build_right_panel(self, parent, theme):
            """Build the right preview panel."""
            # Info card
            self.info_card = tk.Frame(parent, bg=theme["card_bg"],
                                       relief="flat", highlightthickness=1,
                                       highlightbackground=theme["card_border"])
            self.info_card.pack(fill="x", padx=10, pady=(10, 5))

            self.info_title = tk.Label(
                self.info_card, text="📊 Generation Info",
                font=("Segoe UI", 11, "bold"), fg=theme["fg"],
                bg=theme["card_bg"]
            )
            self.info_title.pack(anchor="w", padx=12, pady=(10, 4))

            self.info_label = tk.Label(
                self.info_card, text="", font=("Consolas", 9),
                fg=theme["fg2"], bg=theme["card_bg"],
                justify="left", anchor="w"
            )
            self.info_label.pack(fill="x", padx=12, pady=(0, 10))

            # Waveform preview
            if HAS_MATPLOTLIB:
                preview_card = tk.Frame(parent, bg=theme["card_bg"],
                                         relief="flat", highlightthickness=1,
                                         highlightbackground=theme["card_border"])
                preview_card.pack(fill="both", expand=True, padx=10, pady=5)
                self.preview_card = preview_card

                preview_title = tk.Label(
                    preview_card, text="🌊 Waveform Preview",
                    font=("Segoe UI", 11, "bold"), fg=theme["fg"],
                    bg=theme["card_bg"]
                )
                preview_title.pack(anchor="w", padx=12, pady=(10, 4))
                self.preview_title = preview_title

                self.fig = Figure(figsize=(4, 5), dpi=85)
                self.fig.set_facecolor(theme["plot_bg"])
                self.ax_wave = self.fig.add_subplot(211)
                self.ax_spec = self.fig.add_subplot(212)
                self.fig.tight_layout(pad=3.0)

                self.canvas = FigureCanvasTkAgg(self.fig, master=preview_card)
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
                self.canvas.get_tk_widget().configure(highlightthickness=0)
            else:
                no_mpl = tk.Label(parent, text="📦 Install matplotlib\nfor waveform preview",
                                   font=("Segoe UI", 11), fg=theme["fg2"],
                                   bg=theme["bg"], justify="center")
                no_mpl.pack(padx=20, pady=60)

        def _build_bottom_bar(self, root, theme):
            """Build the bottom status/progress bar."""
            self.bottom_frame = tk.Frame(root, bg=theme["bg2"], height=80)
            self.bottom_frame.pack(fill="x", side="bottom")
            self.bottom_frame.pack_propagate(False)

            # Progress bar (custom canvas)
            prog_frame = tk.Frame(self.bottom_frame, bg=theme["bg2"])
            prog_frame.pack(fill="x", padx=20, pady=(10, 4))

            self.progress_canvas = tk.Canvas(prog_frame, height=12,
                                              bg=theme["progress_bg"],
                                              highlightthickness=0, borderwidth=0)
            self.progress_canvas.pack(fill="x")

            self._draw_progress(0)

            # Status label
            self.status_label = tk.Label(
                self.bottom_frame, textvariable=self.status_var,
                font=("Segoe UI", 10), fg=theme["fg2"],
                bg=theme["bg2"], anchor="w"
            )
            self.status_label.pack(fill="x", padx=20, pady=(0, 8))

        def _draw_progress(self, percentage):
            """Draw the custom progress bar."""
            theme = self._theme()
            self.progress_canvas.delete("all")
            w = self.progress_canvas.winfo_width()
            if w < 10:
                w = 800
            h = 12
            r = 6  # corner radius

            # Background
            self.progress_canvas.create_rectangle(0, 0, w, h, fill=theme["progress_bg"], outline="")

            # Filled portion
            fill_w = max(0, int(w * percentage / 100))
            if fill_w > 0:
                # Gradient-like effect: use accent color
                self.progress_canvas.create_rectangle(0, 0, fill_w, h,
                                                       fill=theme["progress_fg"], outline="")
                # Shine effect
                self.progress_canvas.create_rectangle(0, 0, fill_w, h // 2,
                                                       fill=self._lighten_color(theme["progress_fg"], 40),
                                                       outline="")

        def _lighten_color(self, color, amount):
            color = color.lstrip("#")
            r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            r = min(255, r + amount)
            g = min(255, g + amount)
            b = min(255, b + amount)
            return f"#{r:02x}{g:02x}{b:02x}"

        def _section_label(self, parent, text, theme):
            """Create a styled section label."""
            if text:
                frame = tk.Frame(parent, bg=theme["bg"])
                frame.pack(fill="x", padx=15, pady=(12, 2))
                lbl = tk.Label(frame, text=text, font=("Segoe UI", 12, "bold"),
                               fg=theme["fg"], bg=theme["bg"])
                lbl.pack(side="left")
                divider = tk.Frame(frame, bg=theme["divider"], height=1)
                divider.pack(fill="x", side="bottom", pady=4)

        def _select_custom(self):
            self.mode_var.set("custom")
            self._on_mode_change()

        def _select_tone(self, tone_type):
            self.tone_type_var.set(tone_type)
            self._update_tone_buttons()
            self._update_preview()

        def _select_noise(self, noise_type):
            self.noise_type_var.set(noise_type)
            self._update_noise_buttons()

        def _update_tone_buttons(self):
            theme = self._theme()
            selected = self.tone_type_var.get()
            for tt_value, widgets in self.tone_buttons.items():
                if tt_value == selected:
                    widgets["frame"].configure(highlightbackground=theme["accent"],
                                                highlightthickness=3)
                    widgets["frame"].configure(bg=theme["bg3"])
                    widgets["emoji"].configure(bg=theme["bg3"])
                    widgets["name"].configure(bg=theme["bg3"], fg=theme["accent"])
                    widgets["desc"].configure(bg=theme["bg3"])
                else:
                    widgets["frame"].configure(highlightbackground=theme["card_border"],
                                                highlightthickness=2)
                    widgets["frame"].configure(bg=theme["card_bg"])
                    widgets["emoji"].configure(bg=theme["card_bg"])
                    widgets["name"].configure(bg=theme["card_bg"], fg=theme["fg"])
                    widgets["desc"].configure(bg=theme["card_bg"])

        def _update_noise_buttons(self):
            theme = self._theme()
            selected = self.noise_type_var.get()
            for nt_value, widgets in self.noise_buttons.items():
                if nt_value == selected:
                    widgets["frame"].configure(highlightbackground=theme["accent"],
                                                highlightthickness=3, bg=theme["bg3"])
                    widgets["label"].configure(bg=theme["bg3"], fg=theme["accent"])
                else:
                    widgets["frame"].configure(highlightbackground=theme["card_border"],
                                                highlightthickness=2, bg=theme["card_bg"])
                    widgets["label"].configure(bg=theme["card_bg"], fg=theme["fg"])

        def _on_mode_change(self, *args):
            theme = self._theme()
            mode = self.mode_var.get()

            # Update mode cards
            for card in self.mode_cards:
                card.set_selected(card.mode_name == mode, theme)

            # Update custom card
            if mode == "custom":
                self.custom_card.configure(highlightbackground=theme["accent"],
                                            highlightthickness=3)
                self.beat_entry.entry.configure(state="normal")
            else:
                self.custom_card.configure(highlightbackground=theme["card_border"],
                                            highlightthickness=2)
                if mode in BRAINWAVE_MODES:
                    self.beat_freq_var.set(str(BRAINWAVE_MODES[mode].beat_frequency))
                self.beat_entry.entry.configure(state="disabled")

            self._update_tone_buttons()
            self._update_noise_buttons()
            self._update_info()
            self._update_preview()

        def _update_info(self):
            try:
                config = self._get_config()
                est_size = BrainwaveGenerator.estimate_file_size(config)
                num_samples = int(config.sample_rate * config.duration_seconds)

                if config.mode in BRAINWAVE_MODES:
                    mi = BRAINWAVE_MODES[config.mode]
                    mode_str = f"{mi.emoji} {mi.description}"
                    band_str = f"Band: {mi.band.upper()}"
                else:
                    mode_str = f"⚙️ Custom ({config.beat_frequency} Hz)"
                    band_str = ""

                info = (
                    f"Mode:     {mode_str}\n"
                    f"{band_str}\n"
                    f"Carrier:  {config.carrier_frequency:.0f} Hz\n"
                    f"Beat:     {config.beat_frequency:.1f} Hz\n"
                    f"Tone:     {config.tone_type}\n"
                    f"Duration: {config.duration_seconds:.0f}s "
                    f"({config.duration_seconds / 60:.1f} min)\n"
                    f"Samples:  {num_samples:,}\n"
                    f"Size:     ~{est_size / 1024 / 1024:.2f} MB\n"
                    f"Noise:    {config.noise_type} "
                    f"({config.noise_intensity * 100:.0f}%)"
                )
                self.info_label.configure(text=info)
            except Exception:
                self.info_label.configure(text="Enter valid parameters.")

        def _update_preview(self, *args):
            if not HAS_MATPLOTLIB:
                return
            try:
                config = self._get_config()
                theme = self._theme()

                # Waveform (20ms)
                t, audio = BrainwaveGenerator.get_preview_data(config, preview_seconds=0.02)
                self.ax_wave.clear()
                self.ax_wave.plot(t * 1000, audio, color=theme["plot_line"], linewidth=0.9)
                self.ax_wave.fill_between(t * 1000, audio, alpha=0.15, color=theme["plot_line"])
                self.ax_wave.set_title("Waveform (20ms)", fontsize=10,
                                       color=theme["plot_fg"], fontweight="bold")
                self.ax_wave.set_xlabel("Time (ms)", fontsize=8, color=theme["plot_fg"])
                self.ax_wave.set_ylabel("Amplitude", fontsize=8, color=theme["plot_fg"])
                self.ax_wave.set_facecolor(theme["plot_bg"])
                self.ax_wave.tick_params(colors=theme["plot_fg"], labelsize=7)
                self.ax_wave.set_ylim(-1.1, 1.1)
                self.ax_wave.grid(True, alpha=0.2, color=theme["plot_fg"])
                for spine in self.ax_wave.spines.values():
                    spine.set_color(theme["plot_fg"])
                    spine.set_alpha(0.3)

                # Spectrum (2s for freq resolution)
                t2, audio2 = BrainwaveGenerator.get_preview_data(config, preview_seconds=2.0)
                freqs = np.fft.rfftfreq(len(audio2), 1.0 / config.sample_rate)
                spectrum = np.abs(np.fft.rfft(audio2))
                spectrum = spectrum / (np.max(spectrum) + 1e-10)

                mask = freqs <= 500
                self.ax_spec.clear()
                self.ax_spec.plot(freqs[mask], spectrum[mask],
                                  color=theme["plot_line2"], linewidth=0.9)
                self.ax_spec.fill_between(freqs[mask], spectrum[mask],
                                           alpha=0.15, color=theme["plot_line2"])
                self.ax_spec.set_title("Frequency Spectrum", fontsize=10,
                                       color=theme["plot_fg"], fontweight="bold")
                self.ax_spec.set_xlabel("Frequency (Hz)", fontsize=8, color=theme["plot_fg"])
                self.ax_spec.set_ylabel("Magnitude", fontsize=8, color=theme["plot_fg"])
                self.ax_spec.set_facecolor(theme["plot_bg"])
                self.ax_spec.tick_params(colors=theme["plot_fg"], labelsize=7)
                self.ax_spec.grid(True, alpha=0.2, color=theme["plot_fg"])
                for spine in self.ax_spec.spines.values():
                    spine.set_color(theme["plot_fg"])
                    spine.set_alpha(0.3)

                self.fig.set_facecolor(theme["plot_bg"])
                self.fig.tight_layout(pad=3.0)
                self.canvas.draw()

            except Exception as e:
                logger.debug(f"Preview error: {e}")

        def _get_config(self) -> GenerationConfig:
            try:
                dur_min = float(self.duration_min_var.get() or "0")
            except ValueError:
                dur_min = 0
            try:
                dur_sec = float(self.duration_sec_var.get() or "0")
            except ValueError:
                dur_sec = 0

            total_seconds = dur_min * 60 + dur_sec
            if total_seconds <= 0:
                total_seconds = 300

            mode = self.mode_var.get()
            try:
                beat_freq = float(self.beat_freq_var.get() or "8")
            except ValueError:
                beat_freq = 8.0
            if mode != "custom" and mode in BRAINWAVE_MODES:
                beat_freq = BRAINWAVE_MODES[mode].beat_frequency

            try:
                carrier = float(self.carrier_var.get() or "200")
            except ValueError:
                carrier = 200.0

            try:
                fade = float(self.fade_var.get() or "2.0")
            except ValueError:
                fade = 2.0

            return GenerationConfig(
                mode=mode, beat_frequency=beat_freq,
                carrier_frequency=carrier,
                tone_type=self.tone_type_var.get(),
                duration_seconds=total_seconds,
                volume=self.volume_var.get(),
                sample_rate=int(self.sample_rate_var.get() or "44100"),
                noise_type=self.noise_type_var.get(),
                noise_intensity=self.noise_intensity_var.get(),
                fade_duration=fade,
                output_filename=self.output_file_var.get(),
                output_directory=self.output_dir_var.get(),
                export_flac=self.export_flac_var.get(),
            )

        def _on_generate(self):
            if self.generating:
                messagebox.showinfo("Busy", "Generation already in progress!")
                return
            try:
                config = self._get_config()
                errors = config.validate()
                if errors:
                    messagebox.showerror("⚠️ Validation Error", "\n".join(errors))
                    return
            except ValueError as e:
                messagebox.showerror("⚠️ Input Error", f"Invalid input: {e}")
                return

            self.generating = True
            self.gen_btn.set_enabled(False)
            self.gen_all_btn.set_enabled(False)
            self.progress_var.set(0)
            self._draw_progress(0)
            self.status_var.set("🔄 Generating...")

            def callback(pct, msg):
                self.root.after(0, lambda: (
                    self.progress_var.set(pct),
                    self._draw_progress(pct),
                    self.status_var.set(f"🔄 {msg}")
                ))

            def worker():
                try:
                    filepath = self.generator.generate_and_save(config, progress_callback=callback)
                    self.root.after(0, lambda: (
                        self._draw_progress(100),
                        self.status_var.set(f"✅ Saved: {os.path.abspath(filepath)}"),
                        messagebox.showinfo("✅ Success",
                                            f"Audio saved to:\n{os.path.abspath(filepath)}")
                    ))
                except Exception as e:
                    self.root.after(0, lambda: (
                        self.status_var.set(f"❌ Error: {e}"),
                        messagebox.showerror("❌ Error", str(e))
                    ))
                finally:
                    self.root.after(0, self._generation_done)

            threading.Thread(target=worker, daemon=True).start()

        def _on_generate_all(self):
            if self.generating:
                messagebox.showinfo("Busy", "Generation already in progress!")
                return

            self.generating = True
            self.gen_btn.set_enabled(False)
            self.gen_all_btn.set_enabled(False)
            self.progress_var.set(0)
            self._draw_progress(0)

            def worker():
                modes = list(BRAINWAVE_MODES.keys())
                total = len(modes)
                results = []
                for i, mode_name in enumerate(modes):
                    try:
                        base_config = self._get_config()
                        mode_def = BRAINWAVE_MODES[mode_name]
                        batch_config = GenerationConfig(
                            mode=mode_name,
                            beat_frequency=mode_def.beat_frequency,
                            carrier_frequency=base_config.carrier_frequency,
                            tone_type=base_config.tone_type,
                            duration_seconds=base_config.duration_seconds,
                            volume=base_config.volume,
                            sample_rate=base_config.sample_rate,
                            noise_type=base_config.noise_type,
                            noise_intensity=base_config.noise_intensity,
                            fade_duration=base_config.fade_duration,
                            output_directory=base_config.output_directory,
                            export_flac=base_config.export_flac,
                        )
                        pct = ((i + 1) / total) * 100
                        self.root.after(0, lambda p=pct, m=mode_name, idx=i: (
                            self._draw_progress(p),
                            self.progress_var.set(p),
                            self.status_var.set(
                                f"🔄 Generating {m}... ({idx + 1}/{total})")
                        ))
                        filepath = self.generator.generate_and_save(batch_config)
                        results.append(f"✅ {mode_def.emoji} {mode_name}")
                    except Exception as e:
                        results.append(f"❌ {mode_name}: {e}")

                summary = "\n".join(results)
                self.root.after(0, lambda: (
                    self._draw_progress(100),
                    self.progress_var.set(100),
                    self.status_var.set("✅ Batch generation complete!"),
                    messagebox.showinfo("✅ Batch Complete",
                                        f"Results:\n\n{summary}")
                ))
                self.root.after(0, self._generation_done)

            threading.Thread(target=worker, daemon=True).start()

        def _generation_done(self):
            self.generating = False
            self.gen_btn.set_enabled(True)
            self.gen_all_btn.set_enabled(True)

        def _browse_dir(self):
            d = filedialog.askdirectory(initialdir=self.output_dir_var.get())
            if d:
                self.output_dir_var.set(d)

        def _save_config(self):
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile="brainwave_config.json",
            )
            if filepath:
                try:
                    config = self._get_config()
                    ConfigManager.save(config, filepath)
                    self.status_var.set(f"💾 Config saved to {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        def _load_config(self):
            filepath = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
            )
            if filepath:
                try:
                    config = ConfigManager.load(filepath)
                    self.mode_var.set(config.mode)
                    self.beat_freq_var.set(str(config.beat_frequency))
                    self.carrier_var.set(str(config.carrier_frequency))
                    self.tone_type_var.set(config.tone_type)
                    mins = int(config.duration_seconds // 60)
                    secs = int(config.duration_seconds % 60)
                    self.duration_min_var.set(str(mins))
                    self.duration_sec_var.set(str(secs))
                    self.volume_var.set(config.volume)
                    self.sample_rate_var.set(str(config.sample_rate))
                    self.noise_type_var.set(config.noise_type)
                    self.noise_intensity_var.set(config.noise_intensity)
                    self.fade_var.set(str(config.fade_duration))
                    self.output_dir_var.set(config.output_directory)
                    self.output_file_var.set(config.output_filename)
                    self.export_flac_var.set(config.export_flac)
                    self._on_mode_change()
                    self.status_var.set(f"📂 Config loaded from {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        def _toggle_theme(self):
            self.current_theme = "dark" if self.current_theme == "light" else "light"
            self._apply_theme()
            self._update_preview()

        def _apply_theme(self):
            """Apply theme colors to all widgets recursively."""
            theme = self._theme()

            # Root and main frames
            self.root.configure(bg=theme["bg"])
            self.main_frame.configure(bg=theme["bg"])

            # Header
            self.header_frame.configure(bg=theme["header_bg"])
            self.header_title.configure(bg=theme["header_bg"], fg=theme["header_fg"])
            self.header_subtitle.configure(bg=theme["header_bg"], fg=theme["header_fg"])
            self.theme_btn.configure(bg=theme["header_bg"])
            if self.current_theme == "light":
                self.theme_btn.update_colors(bg="#FFFFFF", fg=theme["header_bg"])
                self.theme_btn._text = "Dark Mode"
                self.theme_btn._emoji = "🌙"
            else:
                self.theme_btn.update_colors(bg="#FFFFFF", fg=theme["header_bg"])
                self.theme_btn._text = "Light Mode"
                self.theme_btn._emoji = "☀️"
            self.theme_btn._draw(self.theme_btn._bg)

            # Left canvas + scrollbar
            self.left_canvas.configure(bg=theme["bg"])
            self.left_scroll_frame.configure(bg=theme["bg"])

            # Right frame
            self.right_frame.configure(bg=theme["bg"])

            # Update mode cards
            for card in self.mode_cards:
                card.update_theme(theme)

            # Custom card
            is_custom = self.mode_var.get() == "custom"
            custom_bg = theme["bg3"] if is_custom else theme["card_bg"]
            self.custom_card.configure(
                bg=custom_bg,
                highlightbackground=theme["accent"] if is_custom else theme["card_border"]
            )
            self.custom_emoji.configure(bg=custom_bg)
            self.custom_label.configure(bg=custom_bg, fg=theme["fg"])

            # Tone buttons
            self._update_tone_buttons()

            # Noise buttons
            self._update_noise_buttons()

            # Styled entries
            for entry_widget in [self.carrier_entry, self.beat_entry,
                                  self.dur_min_entry, self.dur_sec_entry,
                                  self.fade_entry, self.outdir_entry,
                                  self.outfile_entry]:
                entry_widget.update_colors(
                    bg=theme["entry_bg"], fg=theme["entry_fg"],
                    border=theme["entry_border"], accent=theme["accent"],
                    label_fg=theme["fg2"], master_bg=theme["card_bg"]
                )

            # Sample rate menu
            self.sr_menu.configure(
                bg=theme["entry_bg"], fg=theme["entry_fg"],
                activebackground=theme["accent"], activeforeground="#FFFFFF",
                highlightbackground=theme["entry_border"]
            )

            # Volume slider
            self.volume_slider.configure(
                bg=theme["card_bg"], troughcolor=theme["slider_trough"],
                fg=theme["accent"], activebackground=theme["accent"]
            )
            self.vol_display.configure(fg=theme["accent"], bg=theme["card_bg"])

            # Noise slider
            self.noise_slider.configure(
                bg=theme["card_bg"], troughcolor=theme["slider_trough"],
                fg=theme["accent"], activebackground=theme["accent"]
            )
            self.noise_pct_label.configure(fg=theme["accent"], bg=theme["card_bg"])
            self.noise_card.configure(bg=theme["card_bg"],
                                       highlightbackground=theme["card_border"])

            # Output card
            self.out_card.configure(bg=theme["card_bg"],
                                     highlightbackground=theme["card_border"])
            self.flac_cb.configure(bg=theme["card_bg"], fg=theme["fg"],
                                    selectcolor=theme["entry_bg"],
                                    activebackground=theme["card_bg"],
                                    activeforeground=theme["fg"])
            self.flac_var_frame.configure(bg=theme["card_bg"])

            # Buttons
            self.gen_btn.update_colors(theme["btn_primary"], theme["btn_primary_fg"])
            self.gen_btn.configure(bg=theme["bg"])
            self.gen_all_btn.update_colors(theme["btn_success"], theme["btn_success_fg"])
            self.gen_all_btn.configure(bg=theme["bg"])
            self.preview_btn.update_colors(theme["btn_warning"], theme["btn_warning_fg"])
            self.preview_btn.configure(bg=theme["bg"])
            self.save_cfg_btn.update_colors(theme["btn_secondary"], theme["btn_secondary_fg"])
            self.save_cfg_btn.configure(bg=theme["bg"])
            self.load_cfg_btn.update_colors(theme["btn_secondary"], theme["btn_secondary_fg"])
            self.load_cfg_btn.configure(bg=theme["bg"])
            self.browse_btn.update_colors(theme["btn_secondary"], theme["btn_secondary_fg"])
            self.browse_btn.configure(bg=theme["card_bg"])

            # Info card
            self.info_card.configure(bg=theme["card_bg"],
                                      highlightbackground=theme["card_border"])
            self.info_title.configure(bg=theme["card_bg"], fg=theme["fg"])
            self.info_label.configure(bg=theme["card_bg"], fg=theme["fg2"])

            # Preview card
            if HAS_MATPLOTLIB:
                self.preview_card.configure(bg=theme["card_bg"],
                                             highlightbackground=theme["card_border"])
                self.preview_title.configure(bg=theme["card_bg"], fg=theme["fg"])
                self.canvas.get_tk_widget().configure(bg=theme["plot_bg"])

            # Bottom bar
            self.bottom_frame.configure(bg=theme["bg2"])
            self.progress_canvas.configure(bg=theme["progress_bg"])
            self._draw_progress(self.progress_var.get())
            self.status_label.configure(bg=theme["bg2"], fg=theme["fg2"])

            # Recursively update remaining bg for settings frames
            self._update_frame_colors(self.left_scroll_frame, theme)

        def _update_frame_colors(self, widget, theme):
            """Recursively update background colors for frame hierarchy."""
            try:
                widget_type = widget.winfo_class()
                if widget_type == "Frame":
                    current_bg = str(widget.cget("bg"))
                    # Don't override card backgrounds that are specifically set
                    # Only update generic frames
                    pass
            except Exception:
                pass

        def run(self):
            self._update_info()
            self._update_preview()
            self.root.mainloop()

    app = BrainwaveApp()
    app.run()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point. Dispatches to CLI or GUI based on arguments."""
    parser = build_cli_parser()

    if len(sys.argv) == 1:
        logger.info("Launching GUI...")
        run_gui()
        return

    args = parser.parse_args()

    if args.interface == "gui":
        run_gui()
    elif args.interface == "cli":
        run_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()