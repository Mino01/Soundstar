# Starwood Guitar Pro Sound Engine: Technical Specification

**Version**: 1.0  
**Date**: January 2026  
**Author**: Manus AI  
**Repository**: https://github.com/Mino01/Starwood

---

## Executive Summary

This document specifies the architecture and implementation strategy for the **Starwood Guitar Pro Sound Engine**, a next-generation audio rendering system that transforms Guitar Pro tablature files into studio-quality guitar audio. Unlike traditional sample-based approaches (such as Guitar Pro's RSE), Starwood leverages **neural synthesis** and **physical modeling** combined with its proprietary **tonewood transformation technology** to produce authentic, customizable guitar sounds.

The Starwood Sound Engine enables musicians, composers, and producers to:

1. **Render Guitar Pro tabs** with realistic, studio-quality audio
2. **Apply premium tonewood character** (Brazilian Rosewood, Cocobolo, etc.) to any arrangement
3. **Customize articulations** with unprecedented control
4. **Export production-ready audio** for mixing and mastering

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

The Starwood Guitar Pro Sound Engine consists of five primary modules working in concert:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STARWOOD GUITAR PRO SOUND ENGINE                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   GP File   │───▶│  Tab Parser │───▶│  Event Sequencer        │ │
│  │   Input     │    │  (PyGP)     │    │  (MIDI-like events)     │ │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘ │
│                                                     │               │
│                                                     ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              STARWOOD SYNTHESIS CORE                         │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │   │
│  │  │   Tonewood    │  │     DDSP      │  │   Articulation  │  │   │
│  │  │   Character   │──│   Synthesis   │──│    Processor    │  │   │
│  │  │    Module     │  │     Core      │  │                 │  │   │
│  │  └───────────────┘  └───────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                                │
│                                    ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FX CHAIN & MIXER                          │   │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────────┐  │   │
│  │  │ EQ  │──│Comp │──│ Amp │──│ Cab │──│ Rev │──│ Master  │  │   │
│  │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                                │
│                                    ▼                                │
│                          ┌─────────────────┐                        │
│                          │  Audio Output   │                        │
│                          │  (WAV/MP3/FLAC) │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Responsibilities

| Module | Responsibility | Key Technologies |
|--------|----------------|------------------|
| **GP File Input** | Accept Guitar Pro files (.gp3, .gp4, .gp5, .gpx, .gp) | File I/O |
| **Tab Parser** | Parse tablature structure, notes, effects | PyGuitarPro [1] |
| **Event Sequencer** | Convert tab data to time-sequenced events | Custom scheduler |
| **Tonewood Character Module** | Apply premium wood tonal characteristics | Neural embeddings |
| **DDSP Synthesis Core** | Generate audio from synthesis parameters | DDSP, RAVE [2] [3] |
| **Articulation Processor** | Handle guitar-specific playing techniques | Physical modeling |
| **FX Chain & Mixer** | Apply effects and mix tracks | DSP algorithms |
| **Audio Output** | Export final audio in various formats | Audio codecs |

---

## 2. Guitar Pro File Parsing

### 2.1 Supported File Formats

The Starwood Sound Engine supports all major Guitar Pro file formats through the PyGuitarPro library:

| Format | Extension | Version | Support Level |
|--------|-----------|---------|---------------|
| Guitar Pro 3 | .gp3 | 3.x | Full |
| Guitar Pro 4 | .gp4 | 4.x | Full |
| Guitar Pro 5 | .gp5 | 5.x | Full |
| Guitar Pro 6 | .gpx | 6.x | Full |
| Guitar Pro 7/8 | .gp | 7.x/8.x | Full |

### 2.2 Data Model Extraction

The parser extracts the following hierarchical data structure from Guitar Pro files:

```python
class StarwoodTabData:
    """Parsed Guitar Pro file data model."""
    
    # Song metadata
    title: str
    artist: str
    album: str
    tempo: int  # BPM
    
    # Track information
    tracks: List[TrackData]
    
    # Global settings
    master_volume: float
    master_reverb: float

class TrackData:
    """Individual track (instrument) data."""
    
    name: str
    instrument: InstrumentType  # Guitar, Bass, etc.
    tuning: List[int]  # MIDI notes for each string
    capo: int  # Fret position
    strings: int  # Number of strings (6, 7, 8, etc.)
    
    # Sound settings
    channel: ChannelData
    
    # Musical content
    measures: List[MeasureData]

class NoteData:
    """Individual note with all effects."""
    
    fret: int
    string: int
    duration: Duration
    velocity: int
    
    # Articulation effects
    effects: NoteEffects

class NoteEffects:
    """Guitar-specific articulation effects."""
    
    palm_mute: bool
    hammer_on: bool
    pull_off: bool
    slide: Optional[SlideType]
    bend: Optional[BendData]
    vibrato: bool
    harmonic: Optional[HarmonicType]
    let_ring: bool
    staccato: bool
    tremolo_picking: bool
    tapping: bool
    ghost_note: bool
    accent: bool
    dead_note: bool
```

### 2.3 Pitch Calculation

Converting fret/string positions to MIDI pitch requires accounting for tuning and capo:

```python
def fret_string_to_midi(fret: int, string: int, tuning: List[int], capo: int) -> int:
    """
    Convert fret and string position to MIDI note number.
    
    Args:
        fret: Fret number (0 = open string)
        string: String number (0 = highest pitch string)
        tuning: List of MIDI notes for open strings
        capo: Capo position (0 = no capo)
    
    Returns:
        MIDI note number (0-127)
    """
    base_pitch = tuning[string]
    effective_fret = fret + capo if fret > 0 else 0
    return base_pitch + effective_fret
```

Standard guitar tuning (E standard) in MIDI notes:

| String | Note | MIDI Number |
|--------|------|-------------|
| 1 (high E) | E4 | 64 |
| 2 | B3 | 59 |
| 3 | G3 | 55 |
| 4 | D3 | 50 |
| 5 | A2 | 45 |
| 6 (low E) | E2 | 40 |

---

## 3. Event Sequencing

### 3.1 Event Types

The Event Sequencer converts parsed tab data into a time-ordered stream of synthesis events:

```python
@dataclass
class SynthesisEvent:
    """Base class for all synthesis events."""
    timestamp: float  # Seconds from song start
    track_id: int
    string: int

@dataclass
class NoteOnEvent(SynthesisEvent):
    """Note attack event."""
    midi_pitch: int
    velocity: float  # 0.0 - 1.0
    articulation: ArticulationType
    tonewood_params: TonewoodParams

@dataclass
class NoteOffEvent(SynthesisEvent):
    """Note release event."""
    release_type: ReleaseType  # Natural, muted, etc.

@dataclass
class PitchBendEvent(SynthesisEvent):
    """Continuous pitch modification."""
    bend_amount: float  # Semitones

@dataclass
class ArticulationEvent(SynthesisEvent):
    """Mid-note articulation change."""
    articulation: ArticulationType
    intensity: float
```

### 3.2 Timing Conversion

Guitar Pro uses a beat-based timing system that must be converted to absolute time:

```python
def beats_to_seconds(beats: float, tempo: int, time_signature: Tuple[int, int]) -> float:
    """
    Convert beat position to seconds.
    
    Args:
        beats: Number of beats from measure start
        tempo: Beats per minute
        time_signature: (numerator, denominator)
    
    Returns:
        Time in seconds
    """
    beat_duration = 60.0 / tempo
    # Adjust for time signature denominator
    beat_duration *= (4.0 / time_signature[1])
    return beats * beat_duration
```

---

## 4. Starwood Synthesis Core

### 4.1 Tonewood Character Module

The Tonewood Character Module is Starwood's unique differentiator, applying the acoustic characteristics of premium tonewoods to synthesized guitar sounds.

#### 4.1.1 Tonewood Embedding Network

Each tonewood is represented as a learned 256-dimensional embedding vector that captures its acoustic properties:

```python
class TonewoodEmbedding(nn.Module):
    """Neural network for tonewood character encoding."""
    
    def __init__(self, num_tonewoods: int = 20, embedding_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_tonewoods, embedding_dim)
        
        # Acoustic property encoders
        self.density_encoder = nn.Linear(1, 32)
        self.youngs_modulus_encoder = nn.Linear(1, 32)
        self.damping_encoder = nn.Linear(1, 32)
        
        # Fusion layer
        self.fusion = nn.Linear(embedding_dim + 96, embedding_dim)
    
    def forward(self, tonewood_id: int, acoustic_params: Dict[str, float]) -> torch.Tensor:
        """Generate tonewood embedding from ID and acoustic parameters."""
        base_embedding = self.embedding(tonewood_id)
        
        density_feat = self.density_encoder(acoustic_params['density'])
        modulus_feat = self.youngs_modulus_encoder(acoustic_params['youngs_modulus'])
        damping_feat = self.damping_encoder(acoustic_params['damping'])
        
        combined = torch.cat([base_embedding, density_feat, modulus_feat, damping_feat], dim=-1)
        return self.fusion(combined)
```

#### 4.1.2 Supported Tonewoods

| Tonewood | Density (kg/m³) | Character | Best For |
|----------|-----------------|-----------|----------|
| Brazilian Rosewood | 850-950 | Warm, complex overtones | Fingerstyle, jazz |
| Cocobolo | 1000-1200 | Bright, articulate | Lead, rock |
| East Indian Rosewood | 800-900 | Balanced, warm | All-around |
| Honduran Mahogany | 500-600 | Woody, midrange | Blues, folk |
| Koa | 600-700 | Bright, focused | Pop, country |
| Sitka Spruce | 400-450 | Clear, dynamic | Strumming |
| Adirondack Spruce | 380-420 | Powerful, loud | Bluegrass |
| European Spruce | 400-440 | Complex, refined | Classical |

### 4.2 DDSP Synthesis Core

The DDSP (Differentiable Digital Signal Processing) core generates audio using interpretable signal processing components controlled by neural networks [2].

#### 4.2.1 Architecture

```python
class StarwoodDDSPSynthesizer(nn.Module):
    """DDSP-based guitar synthesizer with tonewood conditioning."""
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Control signal predictors
        self.f0_predictor = F0Predictor(input_dim=256, hidden_dim=512)
        self.loudness_predictor = LoudnessPredictor(input_dim=256, hidden_dim=256)
        self.harmonic_predictor = HarmonicPredictor(input_dim=256, num_harmonics=100)
        
        # DDSP synthesizer components
        self.harmonic_synth = HarmonicSynthesizer(sample_rate=sample_rate)
        self.noise_synth = FilteredNoiseSynthesizer(sample_rate=sample_rate)
        self.reverb = TrainableReverb(ir_length=sample_rate * 2)
        
        # Tonewood conditioning
        self.tonewood_conditioner = TonewoodConditioner(embedding_dim=256)
    
    def forward(
        self,
        midi_pitch: torch.Tensor,
        velocity: torch.Tensor,
        duration: torch.Tensor,
        tonewood_embedding: torch.Tensor,
        articulation: torch.Tensor
    ) -> torch.Tensor:
        """
        Synthesize guitar audio from control parameters.
        
        Returns:
            Audio waveform tensor [batch, samples]
        """
        # Condition on tonewood
        conditioned = self.tonewood_conditioner(
            midi_pitch, velocity, tonewood_embedding, articulation
        )
        
        # Predict synthesis parameters
        f0 = self.f0_predictor(conditioned, midi_pitch)
        loudness = self.loudness_predictor(conditioned, velocity)
        harmonics = self.harmonic_predictor(conditioned)
        
        # Generate audio
        harmonic_audio = self.harmonic_synth(f0, loudness, harmonics)
        noise_audio = self.noise_synth(loudness, conditioned)
        
        # Mix and apply reverb
        audio = harmonic_audio + noise_audio
        audio = self.reverb(audio, tonewood_embedding)
        
        return audio
```

#### 4.2.2 String-wise Polyphonic Synthesis

Following the approach of Jonason et al. [4], Starwood implements string-wise synthesis for realistic polyphonic guitar:

```python
class PolyphonicGuitarSynthesizer:
    """Six-string polyphonic guitar synthesizer."""
    
    def __init__(self, num_strings: int = 6):
        self.string_synths = [
            StarwoodDDSPSynthesizer() for _ in range(num_strings)
        ]
        self.string_mixer = StringMixer(num_strings)
    
    def synthesize(self, events: List[SynthesisEvent], tonewood: str) -> np.ndarray:
        """
        Synthesize audio from a list of events.
        
        Each string is synthesized independently, then mixed.
        """
        string_outputs = [[] for _ in range(self.num_strings)]
        
        for event in events:
            string_idx = event.string
            audio = self.string_synths[string_idx].synthesize(event, tonewood)
            string_outputs[string_idx].append((event.timestamp, audio))
        
        # Mix all strings
        return self.string_mixer.mix(string_outputs)
```

### 4.3 Articulation Processor

The Articulation Processor handles guitar-specific playing techniques using a combination of signal processing and physical modeling.

#### 4.3.1 Articulation Implementations

| Articulation | Implementation Approach |
|--------------|------------------------|
| **Palm Mute** | Lowpass filter + increased damping coefficient |
| **Hammer-on/Pull-off** | Reduced attack, legato envelope |
| **Slide** | Continuous pitch interpolation with friction noise |
| **Bend** | Pitch envelope with configurable curve |
| **Vibrato** | Pitch LFO with rate and depth parameters |
| **Natural Harmonic** | Harmonic mode with node position |
| **Artificial Harmonic** | Dual-pitch synthesis (fundamental + harmonic) |
| **Tremolo Picking** | Rapid retriggering with alternating stroke |
| **Tapping** | Reduced velocity, fast attack |
| **Ghost Note** | Very low velocity, muted character |

#### 4.3.2 Palm Mute Implementation

```python
class PalmMuteProcessor:
    """Physical modeling of palm mute technique."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
    def apply(
        self,
        audio: np.ndarray,
        mute_intensity: float = 0.7,
        mute_position: float = 0.1  # Distance from bridge (0-1)
    ) -> np.ndarray:
        """
        Apply palm mute effect to audio.
        
        Args:
            audio: Input audio waveform
            mute_intensity: How hard the palm is pressing (0-1)
            mute_position: Where on the string the palm contacts (0-1)
        
        Returns:
            Palm-muted audio
        """
        # Calculate filter cutoff based on mute intensity
        base_cutoff = 2000  # Hz
        cutoff = base_cutoff * (1 - mute_intensity * 0.8)
        
        # Apply lowpass filter
        sos = signal.butter(4, cutoff, 'low', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Apply damping envelope
        damping_rate = 5 + mute_intensity * 20  # Faster decay with harder mute
        envelope = np.exp(-damping_rate * np.arange(len(audio)) / self.sample_rate)
        
        return filtered * envelope
```

#### 4.3.3 Slide Implementation

```python
class SlideProcessor:
    """Continuous pitch slide between notes."""
    
    def synthesize_slide(
        self,
        start_pitch: int,
        end_pitch: int,
        duration: float,
        slide_type: str = 'linear'  # 'linear', 'exponential', 'legato'
    ) -> np.ndarray:
        """
        Generate audio for a slide between two pitches.
        
        Args:
            start_pitch: Starting MIDI pitch
            end_pitch: Ending MIDI pitch
            duration: Slide duration in seconds
            slide_type: Type of pitch interpolation
        
        Returns:
            Slide audio waveform
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, 1, num_samples)
        
        # Generate pitch curve
        if slide_type == 'linear':
            pitch_curve = start_pitch + (end_pitch - start_pitch) * t
        elif slide_type == 'exponential':
            pitch_curve = start_pitch * np.power(end_pitch / start_pitch, t)
        elif slide_type == 'legato':
            # S-curve for smooth legato slide
            pitch_curve = start_pitch + (end_pitch - start_pitch) * (3*t**2 - 2*t**3)
        
        # Convert to frequency
        freq_curve = 440 * np.power(2, (pitch_curve - 69) / 12)
        
        # Add friction noise characteristic of slides
        friction_noise = self._generate_friction_noise(num_samples, speed=abs(end_pitch - start_pitch))
        
        # Synthesize with varying pitch
        audio = self._synthesize_with_pitch_curve(freq_curve)
        
        return audio + friction_noise * 0.05
```

---

## 5. FX Chain and Mixer

### 5.1 Default FX Chain

The Starwood Sound Engine includes a production-ready FX chain:

```
Input → EQ → Compressor → Amp Sim → Cabinet IR → Delay → Reverb → Master
```

### 5.2 Available Effects

| Category | Effects |
|----------|---------|
| **EQ** | Parametric EQ, Graphic EQ, High/Low Pass |
| **Dynamics** | Compressor, Limiter, Gate |
| **Amp Simulation** | Clean, Crunch, High Gain, Acoustic |
| **Cabinet IRs** | 30+ impulse responses (1x12, 2x12, 4x12, etc.) |
| **Modulation** | Chorus, Flanger, Phaser, Tremolo |
| **Time-based** | Delay, Reverb (Room, Hall, Plate, Spring) |
| **Distortion** | Overdrive, Distortion, Fuzz |

### 5.3 Track Mixing

For multi-track Guitar Pro files, the mixer handles:

- Per-track volume and pan
- Per-track FX sends
- Master bus processing
- Stereo width control

---

## 6. API Design

### 6.1 Python API

```python
from starwood import GuitarProSoundEngine

# Initialize engine
engine = GuitarProSoundEngine(
    sample_rate=48000,
    default_tonewood='brazilian_rosewood'
)

# Load Guitar Pro file
song = engine.load('song.gp5')

# Configure tracks
song.tracks[0].tonewood = 'cocobolo'
song.tracks[0].fx_preset = 'clean_studio'

# Render to audio
audio = engine.render(song)

# Export
engine.export(audio, 'output.wav', format='wav', bit_depth=24)
```

### 6.2 REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/render` | POST | Render GP file to audio |
| `/api/v1/tonewoods` | GET | List available tonewoods |
| `/api/v1/presets` | GET | List FX presets |
| `/api/v1/articulations` | GET | List supported articulations |
| `/api/v1/export` | POST | Export rendered audio |

---

## 7. Performance Considerations

### 7.1 Rendering Speed Targets

| Complexity | Target Speed | Notes |
|------------|--------------|-------|
| Simple (single track, no FX) | 10x real-time | CPU only |
| Medium (2-4 tracks, basic FX) | 5x real-time | CPU only |
| Complex (6+ tracks, full FX) | 2x real-time | CPU only |
| Complex with GPU | 10x real-time | CUDA/Metal |

### 7.2 Optimization Strategies

1. **Batch Processing**: Process multiple notes simultaneously
2. **Caching**: Cache tonewood embeddings and FX parameters
3. **GPU Acceleration**: Use CUDA/Metal for neural network inference
4. **Streaming**: Process audio in chunks for memory efficiency
5. **Parallel Tracks**: Render tracks in parallel threads

---

## 8. Integration with Starwood Ecosystem

### 8.1 Hardware Integration

The Starwood Guitar Pro Sound Engine can work in conjunction with the Starwood Hardware device:

1. **Preview Mode**: Hear arrangements through the hardware transducer
2. **A/B Comparison**: Compare rendered audio with live playing
3. **Hybrid Recording**: Blend rendered and live guitar

### 8.2 DAW Plugin Integration

The engine can be exposed as:

- VST3 plugin (Windows/macOS)
- AU plugin (macOS)
- AAX plugin (Pro Tools)

---

## 9. Conclusion

The Starwood Guitar Pro Sound Engine represents a significant advancement in tablature-to-audio rendering technology. By combining neural synthesis (DDSP), physical modeling, and proprietary tonewood transformation, it delivers studio-quality guitar sounds that surpass traditional sample-based approaches.

Key differentiators:

1. **Tonewood Character**: Apply premium wood characteristics to any arrangement
2. **Neural Synthesis**: High-quality audio without massive sample libraries
3. **Articulation Fidelity**: Realistic handling of all guitar techniques
4. **Production Ready**: Built-in FX chain and mixing capabilities

---

## References

[1] PyGuitarPro Documentation. https://pyguitarpro.readthedocs.io/

[2] Engel, J., et al. "DDSP: Differentiable Digital Signal Processing." ICLR 2020. https://arxiv.org/abs/2001.04643

[3] Caillon, A., & Esling, P. "RAVE: A variational autoencoder for fast and high-quality neural audio synthesis." arXiv 2021. https://arxiv.org/abs/2111.05011

[4] Jonason, N., et al. "DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance from String-wise MIDI Input." arXiv 2023. https://arxiv.org/abs/2309.07658

[5] Ho, K.W., et al. "Guitar Virtual Instrument using Physical Modelling with Collision Simulation." ICMC 2020. https://weonix.github.io/Physical-Guitar-ICMC-Demo/

[6] Impact Soundworks. "Shreddage 3 Stratus." https://impactsoundworks.com/product/shreddage-3-stratus/

[7] Guitar Pro Official Website. https://www.guitar-pro.com/

