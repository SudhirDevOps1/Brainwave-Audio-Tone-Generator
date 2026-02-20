
"""
Brainwave Audio Tone Generator
===============================
A production-level application for generating brainwave entrainment audio tones.
Supports binaural beats, monaural beats, isochronic tones, and pure sine waves.

Dependencies: numpy, scipy, matplotlib
Optional: soundfile (for FLAC export)

Usage:
    CLI Mode: python solution.py cli --mode relax --duration 300 --volume 0.7
    GUI Mode: python solution.py gui
    Default (no args): launches GUI
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

# Configure logging
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
    beat_frequency: float  # Hz
    description: str
    band: str  # delta, theta, alpha, beta, gamma


# Predefined brainwave modes
BRAINWAVE_MODES: Dict[str, BrainwaveMode] = {
    "reading": BrainwaveMode("reading", 10.0, "10 Hz Alpha - Reading Enhancement", "alpha"),
    "study": BrainwaveMode("study", 14.0, "14 Hz Low Beta - Study Aid", "beta"),
    "deep_focus": BrainwaveMode("deep_focus", 16.0, "16 Hz Beta - Deep Focus", "beta"),
    "gamma_focus": BrainwaveMode("gamma_focus", 40.0, "40 Hz Gamma - Peak Focus", "gamma"),
    "relax": BrainwaveMode("relax", 8.0, "8 Hz Alpha - Relaxation", "alpha"),
    "stress_relief": BrainwaveMode("stress_relief", 6.0, "6 Hz Theta - Stress Relief", "theta"),
    "sleep": BrainwaveMode("sleep", 2.0, "2 Hz Delta - Sleep Induction", "delta"),
    "meditation": BrainwaveMode("meditation", 7.0, "7 Hz Theta - Meditation", "theta"),
    "creativity": BrainwaveMode("creativity", 5.0, "5 Hz Theta - Creativity Boost", "theta"),
    "memory_boost": BrainwaveMode("memory_boost", 18.0, "18 Hz Beta - Memory Enhancement", "beta"),
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

        The sine wave is the fundamental building block of all tones.
        f(t) = sin(2π * frequency * t)
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return np.sin(2.0 * np.pi * frequency * t)

    @staticmethod
    def generate_binaural(
        carrier_freq: float,
        beat_freq: float,
        duration: float,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binaural beat as stereo (left, right) channels.

        Binaural beats work by presenting two slightly different frequencies
        to each ear. The brain perceives a third tone at the difference frequency.

        Left ear: carrier_freq
        Right ear: carrier_freq + beat_freq
        Perceived beat: beat_freq Hz
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        left = np.sin(2.0 * np.pi * carrier_freq * t)
        right = np.sin(2.0 * np.pi * (carrier_freq + beat_freq) * t)
        return left, right

    @staticmethod
    def generate_monaural(
        carrier_freq: float,
        beat_freq: float,
        duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Generate monaural beat (amplitude-modulated sine wave).

        Monaural beats are created by mixing two tones externally (in the audio),
        producing a clearly audible amplitude modulation. Unlike binaural beats,
        they don't require headphones.

        f(t) = sin(2π * f_carrier * t) * (0.5 + 0.5 * sin(2π * f_beat * t))

        The modulation envelope oscillates between 0 and 1 at the beat frequency.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        carrier = np.sin(2.0 * np.pi * carrier_freq * t)
        # Modulation envelope: oscillates between 0 and 1
        modulator = 0.5 + 0.5 * np.sin(2.0 * np.pi * beat_freq * t)
        return carrier * modulator

    @staticmethod
    def generate_isochronic(
        carrier_freq: float,
        beat_freq: float,
        duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Generate isochronic tone (sharp on/off pulsing).

        Isochronic tones use evenly spaced, distinct pulses of a single tone.
        The carrier tone is turned on and off at the beat frequency rate.
        This creates a very clear, sharp rhythmic stimulation.

        The pulse is created using a square-wave-like envelope derived from
        a sine wave: when sin(2π * beat * t) >= 0, the tone is ON; otherwise OFF.
        A slight smoothing is applied to reduce harsh clicks.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        carrier = np.sin(2.0 * np.pi * carrier_freq * t)

        # Create sharp pulse envelope using a smoothed square wave
        # We use a high-power cosine to create steep but smooth transitions
        pulse_wave = np.sin(2.0 * np.pi * beat_freq * t)
        # Smooth step: use a sigmoid-like transformation for smoother edges
        # This avoids the harsh clicks of a pure square wave
        steepness = 10.0  # Controls edge sharpness
        envelope = 1.0 / (1.0 + np.exp(-steepness * pulse_wave))

        return carrier * envelope

    @staticmethod
    def generate_white_noise(num_samples: int) -> np.ndarray:
        """
        Generate white noise: equal energy at all frequencies.
        Simply random samples from a uniform or Gaussian distribution.
        """
        return np.random.randn(num_samples)

    @staticmethod
    def generate_pink_noise(num_samples: int) -> np.ndarray:
        """
        Generate pink noise (1/f noise): energy decreases by 3dB per octave.

        Uses the Voss-McCartney algorithm approximation via spectral shaping:
        1. Generate white noise in frequency domain
        2. Apply 1/sqrt(f) filter
        3. Transform back to time domain

        Pink noise sounds more natural and is commonly used in sound therapy.
        """
        # Generate white noise in frequency domain
        white = np.fft.rfft(np.random.randn(num_samples))
        # Create 1/f filter (1/sqrt(f) for amplitude, since power = amplitude^2)
        freqs = np.fft.rfftfreq(num_samples)
        # Avoid division by zero at DC component
        freqs[0] = 1.0
        pink_filter = 1.0 / np.sqrt(freqs)
        pink_filter[0] = 0.0  # Zero DC offset
        # Apply filter and inverse FFT
        pink = np.fft.irfft(white * pink_filter, n=num_samples)
        return pink

    @staticmethod
    def generate_brown_noise(num_samples: int) -> np.ndarray:
        """
        Generate brown (Brownian/red) noise: energy decreases by 6dB per octave.

        Brown noise is generated by integrating white noise (cumulative sum).
        This produces a random walk, which has a 1/f² power spectrum.

        The result is a deep, rumbling sound like a waterfall or strong wind.
        """
        white = np.random.randn(num_samples)
        brown = np.cumsum(white)
        # Normalize to prevent drift from causing extreme values
        brown = brown - np.mean(brown)
        return brown

    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        fade_duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Apply smooth fade-in and fade-out to prevent click artifacts.

        Uses a raised cosine (Hann) fade curve for a perceptually smooth transition:
        fade_in(t) = 0.5 * (1 - cos(π * t / fade_duration))
        fade_out(t) = 0.5 * (1 + cos(π * (t - start) / fade_duration))
        """
        fade_samples = int(fade_duration * sample_rate)
        if fade_samples == 0 or fade_samples * 2 > len(audio):
            return audio

        audio = audio.copy()

        # Raised cosine fade-in
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
        # Raised cosine fade-out
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))

        if audio.ndim == 1:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        elif audio.ndim == 2:
            # Stereo: apply to both channels
            audio[:fade_samples, 0] *= fade_in
            audio[:fade_samples, 1] *= fade_in
            audio[-fade_samples:, 0] *= fade_out
            audio[-fade_samples:, 1] *= fade_out

        return audio

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1.0, 1.0] range to prevent clipping.
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio

    @staticmethod
    def apply_limiter(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Simple hard limiter to prevent distortion.

        Clips any samples exceeding the threshold using soft clipping (tanh).
        This is gentler than hard clipping and reduces harmonic distortion.
        """
        # Soft clipping using hyperbolic tangent
        # Scale so that the threshold maps to tanh(1) ≈ 0.76
        # Then rescale output to reach threshold at the boundary
        return threshold * np.tanh(audio / threshold)

    @staticmethod
    def to_16bit_pcm(audio: np.ndarray) -> np.ndarray:
        """
        Convert floating-point audio [-1.0, 1.0] to 16-bit PCM integers.
        16-bit PCM range: -32768 to 32767
        """
        audio_clipped = np.clip(audio, -1.0, 1.0)
        return (audio_clipped * 32767).astype(np.int16)


# ============================================================
# AUDIO GENERATOR (HIGH-LEVEL)
# ============================================================

class BrainwaveGenerator:
    """
    High-level audio generator that orchestrates the AudioEngine
    to produce complete brainwave entrainment audio files.
    """

    def __init__(self):
        self.engine = AudioEngine()

    def generate(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio based on configuration.

        Returns:
            Tuple of (audio_data as int16 numpy array, sample_rate)
        """
        def report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct:.0f}%] {msg}")

        # Resolve beat frequency from mode
        beat_freq = config.beat_frequency
        if config.mode != "custom" and config.mode in BRAINWAVE_MODES:
            beat_freq = BRAINWAVE_MODES[config.mode].beat_frequency

        sr = config.sample_rate
        dur = config.duration_seconds
        carrier = config.carrier_frequency
        num_samples = int(sr * dur)

        report(5, f"Generating {config.tone_type} tone: carrier={carrier}Hz, beat={beat_freq}Hz, duration={dur}s")

        # ---- Step 1: Generate base tone ----
        tone_type = ToneType(config.tone_type)

        if tone_type == ToneType.PURE:
            # Pure sine at carrier frequency (mono, will be duplicated to stereo)
            mono = self.engine.generate_sine(carrier, dur, sr)
            audio = np.column_stack([mono, mono])  # Stereo
            report(30, "Pure sine wave generated.")

        elif tone_type == ToneType.BINAURAL:
            left, right = self.engine.generate_binaural(carrier, beat_freq, dur, sr)
            audio = np.column_stack([left, right])
            report(30, f"Binaural beat generated: L={carrier}Hz, R={carrier + beat_freq}Hz")

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

        # ---- Step 2: Generate and mix noise ----
        noise_type = NoiseType(config.noise_type)
        if noise_type != NoiseType.NONE and config.noise_intensity > 0:
            report(45, f"Generating {noise_type.value} noise at {config.noise_intensity * 100:.0f}% intensity...")

            if noise_type == NoiseType.WHITE:
                noise = self.engine.generate_white_noise(num_samples)
            elif noise_type == NoiseType.PINK:
                noise = self.engine.generate_pink_noise(num_samples)
            elif noise_type == NoiseType.BROWN:
                noise = self.engine.generate_brown_noise(num_samples)
            else:
                noise = np.zeros(num_samples)

            # Normalize noise
            noise = self.engine.normalize(noise)

            # Mix noise into both channels
            noise_level = config.noise_intensity
            tone_level = 1.0 - noise_level * 0.5  # Reduce tone slightly when noise is added
            noise_stereo = np.column_stack([noise, noise])
            audio = tone_level * audio + noise_level * noise_stereo

            report(60, "Noise mixed into audio.")
        else:
            report(60, "No background noise.")

        # ---- Step 3: Apply volume ----
        audio = audio * config.volume
        report(70, f"Volume applied: {config.volume}")

        # ---- Step 4: Normalize ----
        audio = self.engine.normalize(audio)
        report(75, "Audio normalized.")

        # ---- Step 5: Apply limiter ----
        audio = self.engine.apply_limiter(audio, threshold=0.95)
        report(80, "Limiter applied.")

        # ---- Step 6: Apply fade-in/fade-out ----
        if config.fade_duration > 0:
            audio = self.engine.apply_fade(audio, config.fade_duration, sr)
            report(85, f"Fade applied: {config.fade_duration}s in/out.")

        # ---- Step 7: Final normalization and convert to 16-bit PCM ----
        audio = self.engine.normalize(audio) * config.volume
        audio_pcm = self.engine.to_16bit_pcm(audio)
        report(95, "Converted to 16-bit PCM.")

        return audio_pcm, sr

    def save_wav(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        filepath: str,
    ) -> str:
        """Save audio data as WAV file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        wavfile.write(filepath, sample_rate, audio_data)
        file_size = os.path.getsize(filepath)
        logger.info(f"WAV saved: {filepath} ({file_size / 1024 / 1024:.2f} MB)")
        return filepath

    def save_flac(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        filepath: str,
    ) -> Optional[str]:
        """Save audio data as FLAC file (requires soundfile library)."""
        try:
            import soundfile as sf
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            # Convert int16 back to float for soundfile
            audio_float = audio_data.astype(np.float32) / 32767.0
            sf.write(filepath, audio_float, sample_rate, format="FLAC")
            file_size = os.path.getsize(filepath)
            logger.info(f"FLAC saved: {filepath} ({file_size / 1024 / 1024:.2f} MB)")
            return filepath
        except ImportError:
            logger.warning("soundfile library not available. FLAC export skipped.")
            return None

    def generate_and_save(
        self,
        config: GenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """Generate audio and save to file. Returns the output filepath."""
        # Validate
        errors = config.validate()
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))

        # Determine output filename
        if not config.output_filename:
            mode_name = config.mode if config.mode != "custom" else f"custom_{config.beat_frequency}hz"
            config.output_filename = f"brainwave_{mode_name}_{config.tone_type}_{int(config.duration_seconds)}s.wav"

        filepath = os.path.join(config.output_directory, config.output_filename)

        # Generate
        audio_data, sr = self.generate(config, progress_callback)

        # Save WAV
        self.save_wav(audio_data, sr, filepath)

        # Optionally save FLAC
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
        channels = 2  # Always stereo
        bytes_per_sample = 2  # 16-bit
        header_size = 44  # Standard WAV header
        return header_size + (num_samples * channels * bytes_per_sample)

    @staticmethod
    def get_preview_data(
        config: GenerationConfig, preview_seconds: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a short preview for waveform display. Returns (time_array, audio_mono)."""
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
            audio = (left + right) / 2.0  # Mix to mono for preview
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
        description="Brainwave Audio Tone Generator",
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
    cli_parser.add_argument(
        "--mode", "-m",
        choices=mode_choices,
        default="relax",
        help="Brainwave mode (default: relax)",
    )
    cli_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=300,
        help="Duration in seconds (default: 300)",
    )
    cli_parser.add_argument(
        "--duration-minutes",
        type=float,
        default=None,
        help="Duration in minutes (overrides --duration)",
    )
    cli_parser.add_argument(
        "--volume", "-v",
        type=float,
        default=0.7,
        help="Volume 0.0-1.0 (default: 0.7)",
    )
    cli_parser.add_argument(
        "--carrier", "-c",
        type=float,
        default=200.0,
        help="Carrier/base frequency in Hz (default: 200)",
    )
    cli_parser.add_argument(
        "--beat-freq",
        type=float,
        default=None,
        help="Custom beat frequency in Hz (used with --mode custom)",
    )
    cli_parser.add_argument(
        "--tone-type", "-t",
        choices=[t.value for t in ToneType],
        default="binaural",
        help="Tone type (default: binaural)",
    )
    cli_parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[22050, 44100, 48000, 96000],
        default=44100,
        help="Sample rate (default: 44100)",
    )
    cli_parser.add_argument(
        "--noise",
        choices=[n.value for n in NoiseType],
        default="none",
        help="Background noise type (default: none)",
    )
    cli_parser.add_argument(
        "--noise-intensity",
        type=float,
        default=0.3,
        help="Noise intensity 0.0-1.0 (default: 0.3)",
    )
    cli_parser.add_argument(
        "--fade",
        type=float,
        default=2.0,
        help="Fade in/out duration in seconds (default: 2.0)",
    )
    cli_parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output filename",
    )
    cli_parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_audio",
        help="Output directory (default: generated_audio)",
    )
    cli_parser.add_argument(
        "--flac",
        action="store_true",
        help="Also export as FLAC (requires soundfile)",
    )
    cli_parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all predefined modes",
    )
    cli_parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Save configuration to JSON file",
    )
    cli_parser.add_argument(
        "--load-config",
        type=str,
        default=None,
        help="Load configuration from JSON file",
    )

    # GUI subcommand
    subparsers.add_parser("gui", help="Graphical user interface")

    return parser


def run_cli(args) -> None:
    """Execute CLI mode."""
    generator = BrainwaveGenerator()

    # Load config from file if specified
    if args.load_config:
        config = ConfigManager.load(args.load_config)
        logger.info("Using loaded configuration.")
    else:
        duration = args.duration
        if args.duration_minutes is not None:
            duration = args.duration_minutes * 60.0

        beat_freq = args.beat_freq if args.beat_freq is not None else 8.0
        if args.mode != "custom" and args.mode in BRAINWAVE_MODES:
            beat_freq = BRAINWAVE_MODES[args.mode].beat_frequency

        config = GenerationConfig(
            mode=args.mode,
            beat_frequency=beat_freq,
            carrier_frequency=args.carrier,
            tone_type=args.tone_type,
            duration_seconds=duration,
            volume=args.volume,
            sample_rate=args.sample_rate,
            noise_type=args.noise,
            noise_intensity=args.noise_intensity,
            fade_duration=args.fade,
            output_filename=args.output,
            output_directory=args.output_dir,
            export_flac=args.flac,
        )

    # Save config if requested
    if args.save_config:
        ConfigManager.save(config, args.save_config)

    # Generate all modes
    if args.generate_all:
        logger.info("=" * 60)
        logger.info("BATCH GENERATION: All predefined modes")
        logger.info("=" * 60)
        for mode_name, mode_def in BRAINWAVE_MODES.items():
            batch_config = GenerationConfig(
                mode=mode_name,
                beat_frequency=mode_def.beat_frequency,
                carrier_frequency=config.carrier_frequency,
                tone_type=config.tone_type,
                duration_seconds=config.duration_seconds,
                volume=config.volume,
                sample_rate=config.sample_rate,
                noise_type=config.noise_type,
                noise_intensity=config.noise_intensity,
                fade_duration=config.fade_duration,
                output_directory=config.output_directory,
                export_flac=config.export_flac,
            )
            try:
                filepath = generator.generate_and_save(batch_config)
                logger.info(f"✓ {mode_name}: {filepath}")
            except Exception as e:
                logger.error(f"✗ {mode_name}: {e}")
        logger.info("Batch generation complete.")
        return

    # Single generation
    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Validation error: {err}")
        sys.exit(1)

    # Display info
    estimated_size = BrainwaveGenerator.estimate_file_size(config)
    num_samples = int(config.sample_rate * config.duration_seconds)

    logger.info("=" * 60)
    logger.info("BRAINWAVE AUDIO GENERATOR")
    logger.info("=" * 60)

    if config.mode in BRAINWAVE_MODES:
        mode_info = BRAINWAVE_MODES[config.mode]
        logger.info(f"Mode: {mode_info.description}")
        logger.info(f"Band: {mode_info.band.upper()}")
    else:
        logger.info(f"Mode: Custom ({config.beat_frequency} Hz)")

    logger.info(f"Tone Type: {config.tone_type}")
    logger.info(f"Carrier Frequency: {config.carrier_frequency} Hz")
    logger.info(f"Beat Frequency: {config.beat_frequency} Hz")
    logger.info(f"Duration: {config.duration_seconds:.1f}s ({config.duration_seconds / 60:.1f} min)")
    logger.info(f"Volume: {config.volume}")
    logger.info(f"Sample Rate: {config.sample_rate} Hz")
    logger.info(f"Total Samples: {num_samples:,}")
    logger.info(f"Estimated File Size: {estimated_size / 1024 / 1024:.2f} MB")
    logger.info(f"Noise: {config.noise_type} ({config.noise_intensity * 100:.0f}%)")
    logger.info(f"Fade: {config.fade_duration}s")
    logger.info("-" * 60)

    filepath = generator.generate_and_save(config)
    logger.info("=" * 60)
    logger.info(f"Output: {os.path.abspath(filepath)}")
    logger.info("Generation complete!")


# ============================================================
# GRAPHICAL USER INTERFACE (Tkinter)
# ============================================================

def run_gui() -> None:
    """Launch the Tkinter GUI."""
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        logger.warning("matplotlib not available. Waveform preview disabled.")

    class BrainwaveApp:
        """Main GUI Application."""

        # Theme definitions
        THEMES = {
            "light": {
                "bg": "#f0f0f0",
                "fg": "#000000",
                "entry_bg": "#ffffff",
                "entry_fg": "#000000",
                "frame_bg": "#e8e8e8",
                "accent": "#4a90d9",
                "button_bg": "#4a90d9",
                "button_fg": "#ffffff",
                "status_bg": "#d8d8d8",
                "plot_bg": "#ffffff",
                "plot_fg": "#000000",
            },
            "dark": {
                "bg": "#2d2d2d",
                "fg": "#e0e0e0",
                "entry_bg": "#3d3d3d",
                "entry_fg": "#e0e0e0",
                "frame_bg": "#353535",
                "accent": "#5ba0e0",
                "button_bg": "#5ba0e0",
                "button_fg": "#ffffff",
                "status_bg": "#252525",
                "plot_bg": "#2d2d2d",
                "plot_fg": "#e0e0e0",
            },
        }

        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Brainwave Audio Generator")
            self.root.geometry("920x860")
            self.root.minsize(860, 780)

            self.generator = BrainwaveGenerator()
            self.current_theme = "light"
            self.generating = False

            # Variables
            self.mode_var = tk.StringVar(value="relax")
            self.tone_type_var = tk.StringVar(value="binaural")
            self.duration_min_var = tk.StringVar(value="5")
            self.duration_sec_var = tk.StringVar(value="0")
            self.carrier_var = tk.StringVar(value="200")
            self.volume_var = tk.StringVar(value="0.7")
            self.beat_freq_var = tk.StringVar(value="8.0")
            self.sample_rate_var = tk.StringVar(value="44100")
            self.noise_type_var = tk.StringVar(value="none")
            self.noise_intensity_var = tk.DoubleVar(value=0.3)
            self.fade_var = tk.StringVar(value="2.0")
            self.output_dir_var = tk.StringVar(value="generated_audio")
            self.output_file_var = tk.StringVar(value="")
            self.export_flac_var = tk.BooleanVar(value=False)
            self.progress_var = tk.DoubleVar(value=0.0)
            self.status_var = tk.StringVar(value="Ready.")

            self._build_ui()
            self._apply_theme()
            self._on_mode_change()

        def _build_ui(self):
            """Build the complete user interface."""
            root = self.root

            # Style
            self.style = ttk.Style()

            # Main container with scrollbar potential
            main_frame = ttk.Frame(root, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # ---- Top Bar: Title + Theme Toggle ----
            top_bar = ttk.Frame(main_frame)
            top_bar.pack(fill=tk.X, pady=(0, 8))

            title_label = ttk.Label(top_bar, text="🧠 Brainwave Audio Generator", font=("Helvetica", 16, "bold"))
            title_label.pack(side=tk.LEFT)

            self.theme_btn = ttk.Button(top_bar, text="🌙 Dark Mode", command=self._toggle_theme, width=14)
            self.theme_btn.pack(side=tk.RIGHT)

            # ---- Left + Right panes ----
            panes = ttk.Frame(main_frame)
            panes.pack(fill=tk.BOTH, expand=True)

            left_pane = ttk.Frame(panes)
            left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

            right_pane = ttk.Frame(panes)
            right_pane.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

            # ======== LEFT PANE ========

            # --- Mode Selection ---
            mode_frame = ttk.LabelFrame(left_pane, text="Mode", padding=8)
            mode_frame.pack(fill=tk.X, pady=4)

            mode_choices = list(BRAINWAVE_MODES.keys()) + ["custom"]
            self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, values=mode_choices, state="readonly", width=20)
            self.mode_combo.pack(side=tk.LEFT, padx=5)
            self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())

            self.mode_desc_label = ttk.Label(mode_frame, text="", font=("Helvetica", 9, "italic"))
            self.mode_desc_label.pack(side=tk.LEFT, padx=10)

            # --- Tone Type ---
            tone_frame = ttk.LabelFrame(left_pane, text="Tone Type", padding=8)
            tone_frame.pack(fill=tk.X, pady=4)

            for tt in ToneType:
                rb = ttk.Radiobutton(tone_frame, text=tt.value.capitalize(), variable=self.tone_type_var, value=tt.value,
                                     command=self._update_preview)
                rb.pack(side=tk.LEFT, padx=10)

            # --- Frequency Settings ---
            freq_frame = ttk.LabelFrame(left_pane, text="Frequency Settings", padding=8)
            freq_frame.pack(fill=tk.X, pady=4)

            ttk.Label(freq_frame, text="Carrier (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.carrier_entry = ttk.Entry(freq_frame, textvariable=self.carrier_var, width=10)
            self.carrier_entry.grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(freq_frame, text="Beat Freq (Hz):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
            self.beat_freq_entry = ttk.Entry(freq_frame, textvariable=self.beat_freq_var, width=10)
            self.beat_freq_entry.grid(row=0, column=3, padx=5, pady=2)

            # --- Duration & Volume ---
            dv_frame = ttk.LabelFrame(left_pane, text="Duration & Volume", padding=8)
            dv_frame.pack(fill=tk.X, pady=4)

            ttk.Label(dv_frame, text="Minutes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(dv_frame, textvariable=self.duration_min_var, width=6).grid(row=0, column=1, padx=5, pady=2)

            ttk.Label(dv_frame, text="Seconds:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(dv_frame, textvariable=self.duration_sec_var, width=6).grid(row=0, column=3, padx=5, pady=2)

            ttk.Label(dv_frame, text="Volume (0-1):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(dv_frame, textvariable=self.volume_var, width=6).grid(row=0, column=5, padx=5, pady=2)

            ttk.Label(dv_frame, text="Fade (s):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(dv_frame, textvariable=self.fade_var, width=6).grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(dv_frame, text="Sample Rate:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
            sr_combo = ttk.Combobox(dv_frame, textvariable=self.sample_rate_var,
                                    values=["22050", "44100", "48000", "96000"], state="readonly", width=8)
            sr_combo.grid(row=1, column=3, padx=5, pady=2)

            # --- Noise ---
            noise_frame = ttk.LabelFrame(left_pane, text="Background Noise", padding=8)
            noise_frame.pack(fill=tk.X, pady=4)

            ttk.Label(noise_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5)
            noise_combo = ttk.Combobox(noise_frame, textvariable=self.noise_type_var,
                                       values=[n.value for n in NoiseType], state="readonly", width=10)
            noise_combo.grid(row=0, column=1, padx=5)

            ttk.Label(noise_frame, text="Intensity:").grid(row=0, column=2, sticky=tk.W, padx=5)
            self.noise_slider = ttk.Scale(noise_frame, from_=0.0, to=1.0, variable=self.noise_intensity_var,
                                          orient=tk.HORIZONTAL, length=150)
            self.noise_slider.grid(row=0, column=3, padx=5)

            self.noise_pct_label = ttk.Label(noise_frame, text="30%")
            self.noise_pct_label.grid(row=0, column=4, padx=5)
            self.noise_intensity_var.trace_add("write", lambda *a: self.noise_pct_label.configure(
                text=f"{self.noise_intensity_var.get() * 100:.0f}%"))

            # --- Output ---
            out_frame = ttk.LabelFrame(left_pane, text="Output", padding=8)
            out_frame.pack(fill=tk.X, pady=4)

            ttk.Label(out_frame, text="Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(out_frame, textvariable=self.output_dir_var, width=30).grid(row=0, column=1, padx=5, pady=2)
            ttk.Button(out_frame, text="Browse", command=self._browse_dir, width=8).grid(row=0, column=2, padx=5, pady=2)

            ttk.Label(out_frame, text="Filename:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(out_frame, textvariable=self.output_file_var, width=30).grid(row=1, column=1, padx=5, pady=2)
            ttk.Label(out_frame, text="(auto if empty)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

            ttk.Checkbutton(out_frame, text="Also export FLAC", variable=self.export_flac_var).grid(
                row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

            # --- Info Display ---
            self.info_label = ttk.Label(left_pane, text="", font=("Courier", 9), justify=tk.LEFT)
            self.info_label.pack(fill=tk.X, pady=4)

            # --- Buttons ---
            btn_frame = ttk.Frame(left_pane)
            btn_frame.pack(fill=tk.X, pady=6)

            self.generate_btn = ttk.Button(btn_frame, text="▶ Generate", command=self._on_generate, width=16)
            self.generate_btn.pack(side=tk.LEFT, padx=5)

            self.generate_all_btn = ttk.Button(btn_frame, text="▶ Generate All Modes", command=self._on_generate_all, width=20)
            self.generate_all_btn.pack(side=tk.LEFT, padx=5)

            ttk.Button(btn_frame, text="Preview", command=self._update_preview, width=10).pack(side=tk.LEFT, padx=5)

            ttk.Button(btn_frame, text="Save Config", command=self._save_config, width=12).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Load Config", command=self._load_config, width=12).pack(side=tk.LEFT, padx=5)

            # --- Progress ---
            self.progress_bar = ttk.Progressbar(left_pane, variable=self.progress_var, maximum=100, length=400)
            self.progress_bar.pack(fill=tk.X, pady=4)

            # --- Status ---
            self.status_label = ttk.Label(left_pane, textvariable=self.status_var, font=("Helvetica", 10),
                                          relief=tk.SUNKEN, anchor=tk.W, padding=5)
            self.status_label.pack(fill=tk.X, pady=4)

            # ======== RIGHT PANE: Waveform Preview ========
            if HAS_MATPLOTLIB:
                preview_frame = ttk.LabelFrame(right_pane, text="Waveform Preview", padding=5)
                preview_frame.pack(fill=tk.BOTH, expand=True)

                self.fig = Figure(figsize=(3.5, 6), dpi=90)
                self.ax_wave = self.fig.add_subplot(211)
                self.ax_spec = self.fig.add_subplot(212)
                self.fig.tight_layout(pad=2.0)

                self.canvas = FigureCanvasTkAgg(self.fig, master=preview_frame)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                placeholder = ttk.Label(right_pane, text="Install matplotlib\nfor waveform preview",
                                       font=("Helvetica", 10), justify=tk.CENTER)
                placeholder.pack(padx=20, pady=40)

            self._update_info()

        def _get_config(self) -> GenerationConfig:
            """Build GenerationConfig from GUI inputs."""
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
            beat_freq = float(self.beat_freq_var.get() or "8")
            if mode != "custom" and mode in BRAINWAVE_MODES:
                beat_freq = BRAINWAVE_MODES[mode].beat_frequency

            return GenerationConfig(
                mode=mode,
                beat_frequency=beat_freq,
                carrier_frequency=float(self.carrier_var.get() or "200"),
                tone_type=self.tone_type_var.get(),
                duration_seconds=total_seconds,
                volume=float(self.volume_var.get() or "0.7"),
                sample_rate=int(self.sample_rate_var.get() or "44100"),
                noise_type=self.noise_type_var.get(),
                noise_intensity=self.noise_intensity_var.get(),
                fade_duration=float(self.fade_var.get() or "2.0"),
                output_filename=self.output_file_var.get(),
                output_directory=self.output_dir_var.get(),
                export_flac=self.export_flac_var.get(),
            )

        def _on_mode_change(self, *args):
            """Handle mode selection change."""
            mode = self.mode_var.get()
            if mode in BRAINWAVE_MODES:
                info = BRAINWAVE_MODES[mode]
                self.mode_desc_label.configure(text=info.description)
                self.beat_freq_var.set(str(info.beat_frequency))
                self.beat_freq_entry.configure(state="disabled")
            else:
                self.mode_desc_label.configure(text="Custom frequency")
                self.beat_freq_entry.configure(state="normal")

            self._update_info()
            self._update_preview()

        def _update_info(self):
            """Update the info display label."""
            try:
                config = self._get_config()
                est_size = BrainwaveGenerator.estimate_file_size(config)
                num_samples = int(config.sample_rate * config.duration_seconds)
                info_text = (
                    f"Beat: {config.beat_frequency:.1f} Hz | Carrier: {config.carrier_frequency:.0f} Hz\n"
                    f"Samples: {num_samples:,} | Est. Size: {est_size / 1024 / 1024:.2f} MB\n"
                    f"Duration: {config.duration_seconds:.0f}s ({config.duration_seconds / 60:.1f} min)"
                )
                self.info_label.configure(text=info_text)
            except (ValueError, Exception):
                self.info_label.configure(text="Enter valid parameters.")

        def _update_preview(self, *args):
            """Update the waveform preview plot."""
            if not HAS_MATPLOTLIB:
                return

            try:
                config = self._get_config()

                # Generate preview data (20ms for waveform view)
                t, audio = BrainwaveGenerator.get_preview_data(config, preview_seconds=0.02)

                theme = self.THEMES[self.current_theme]

                # Waveform plot
                self.ax_wave.clear()
                self.ax_wave.plot(t * 1000, audio, color=theme["accent"], linewidth=0.8)
                self.ax_wave.set_title("Waveform (20ms)", fontsize=9, color=theme["plot_fg"])
                self.ax_wave.set_xlabel("Time (ms)", fontsize=8, color=theme["plot_fg"])
                self.ax_wave.set_ylabel("Amplitude", fontsize=8, color=theme["plot_fg"])
                self.ax_wave.set_facecolor(theme["plot_bg"])
                self.ax_wave.tick_params(colors=theme["plot_fg"], labelsize=7)
                self.ax_wave.set_ylim(-1.1, 1.1)

                # Spectrum plot (2 seconds of data for frequency resolution)
                t2, audio2 = BrainwaveGenerator.get_preview_data(config, preview_seconds=2.0)
                freqs = np.fft.rfftfreq(len(audio2), 1.0 / config.sample_rate)
                spectrum = np.abs(np.fft.rfft(audio2))
                spectrum = spectrum / (np.max(spectrum) + 1e-10)

                # Only show up to 500 Hz for readability
                mask = freqs <= 500
                self.ax_spec.clear()
                self.ax_spec.plot(freqs[mask], spectrum[mask], color=theme["accent"], linewidth=0.8)
                self.ax_spec.set_title("Frequency Spectrum", fontsize=9, color=theme["plot_fg"])
                self.ax_spec.set_xlabel("Frequency (Hz)", fontsize=8, color=theme["plot_fg"])
                self.ax_spec.set_ylabel("Magnitude", fontsize=8, color=theme["plot_fg"])
                self.ax_spec.set_facecolor(theme["plot_bg"])
                self.ax_spec.tick_params(colors=theme["plot_fg"], labelsize=7)

                self.fig.set_facecolor(theme["plot_bg"])
                self.fig.tight_layout(pad=2.0)
                self.canvas.draw()

            except Exception as e:
                logger.debug(f"Preview error: {e}")

        def _on_generate(self):
            """Start generation in a background thread."""
            if self.generating:
                messagebox.showinfo("Busy", "Generation already in progress.")
                return

            try:
                config = self._get_config()
                errors = config.validate()
                if errors:
                    messagebox.showerror("Validation Error", "\n".join(errors))
                    return
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid input: {e}")
                return

            self.generating = True
            self.generate_btn.configure(state="disabled")
            self.generate_all_btn.configure(state="disabled")
            self.progress_var.set(0)
            self.status_var.set("Generating...")

            def callback(pct, msg):
                self.root.after(0, lambda: self.progress_var.set(pct))
                self.root.after(0, lambda: self.status_var.set(msg))

            def worker():
                try:
                    filepath = self.generator.generate_and_save(config, progress_callback=callback)
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", f"Audio saved to:\n{os.path.abspath(filepath)}"))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                finally:
                    self.root.after(0, self._generation_done)

            threading.Thread(target=worker, daemon=True).start()

        def _on_generate_all(self):
            """Generate all predefined modes."""
            if self.generating:
                messagebox.showinfo("Busy", "Generation already in progress.")
                return

            self.generating = True
            self.generate_btn.configure(state="disabled")
            self.generate_all_btn.configure(state="disabled")
            self.progress_var.set(0)

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
                        pct = ((i) / total) * 100
                        self.root.after(0, lambda p=pct, m=mode_name: (
                            self.progress_var.set(p),
                            self.status_var.set(f"Generating {m}... ({i + 1}/{total})")
                        ))
                        filepath = self.generator.generate_and_save(batch_config)
                        results.append(f"✓ {mode_name}")
                    except Exception as e:
                        results.append(f"✗ {mode_name}: {e}")

                summary = "\n".join(results)
                self.root.after(0, lambda: (
                    self.progress_var.set(100),
                    self.status_var.set("Batch generation complete!"),
                    messagebox.showinfo("Batch Complete", f"Results:\n\n{summary}")
                ))
                self.root.after(0, self._generation_done)

            threading.Thread(target=worker, daemon=True).start()

        def _generation_done(self):
            """Reset UI after generation completes."""
            self.generating = False
            self.generate_btn.configure(state="normal")
            self.generate_all_btn.configure(state="normal")

        def _browse_dir(self):
            """Open directory browser."""
            d = filedialog.askdirectory(initialdir=self.output_dir_var.get())
            if d:
                self.output_dir_var.set(d)

        def _save_config(self):
            """Save current configuration to JSON."""
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile="brainwave_config.json",
            )
            if filepath:
                try:
                    config = self._get_config()
                    ConfigManager.save(config, filepath)
                    self.status_var.set(f"Config saved to {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        def _load_config(self):
            """Load configuration from JSON."""
            filepath = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
            )
            if filepath:
                try:
                    config = ConfigManager.load(filepath)
                    # Update all GUI fields
                    self.mode_var.set(config.mode)
                    self.beat_freq_var.set(str(config.beat_frequency))
                    self.carrier_var.set(str(config.carrier_frequency))
                    self.tone_type_var.set(config.tone_type)
                    mins = int(config.duration_seconds // 60)
                    secs = int(config.duration_seconds % 60)
                    self.duration_min_var.set(str(mins))
                    self.duration_sec_var.set(str(secs))
                    self.volume_var.set(str(config.volume))
                    self.sample_rate_var.set(str(config.sample_rate))
                    self.noise_type_var.set(config.noise_type)
                    self.noise_intensity_var.set(config.noise_intensity)
                    self.fade_var.set(str(config.fade_duration))
                    self.output_dir_var.set(config.output_directory)
                    self.output_file_var.set(config.output_filename)
                    self.export_flac_var.set(config.export_flac)
                    self._on_mode_change()
                    self.status_var.set(f"Config loaded from {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        def _toggle_theme(self):
            """Toggle between light and dark themes."""
            self.current_theme = "dark" if self.current_theme == "light" else "light"
            self._apply_theme()
            self._update_preview()

        def _apply_theme(self):
            """Apply the current theme to all widgets."""
            theme = self.THEMES[self.current_theme]

            # Update ttk style
            self.style.configure(".", background=theme["bg"], foreground=theme["fg"])
            self.style.configure("TFrame", background=theme["bg"])
            self.style.configure("TLabel", background=theme["bg"], foreground=theme["fg"])
            self.style.configure("TLabelframe", background=theme["bg"], foreground=theme["fg"])
            self.style.configure("TLabelframe.Label", background=theme["bg"], foreground=theme["fg"])
            self.style.configure("TButton", background=theme["button_bg"])
            self.style.configure("TRadiobutton", background=theme["bg"], foreground=theme["fg"])
            self.style.configure("TCheckbutton", background=theme["bg"], foreground=theme["fg"])
            self.style.configure("TEntry", fieldbackground=theme["entry_bg"], foreground=theme["entry_fg"])
            self.style.configure("TCombobox", fieldbackground=theme["entry_bg"], foreground=theme["entry_fg"])

            self.root.configure(bg=theme["bg"])

            # Update theme button text
            if self.current_theme == "light":
                self.theme_btn.configure(text="🌙 Dark Mode")
            else:
                self.theme_btn.configure(text="☀️ Light Mode")

        def run(self):
            """Start the GUI main loop."""
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

    # If no arguments provided, default to GUI
    if len(sys.argv) == 1:
        logger.info("No arguments provided. Launching GUI...")
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