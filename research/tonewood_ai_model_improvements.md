# Tonewood AI Model Improvements Research

## Research Date: January 9, 2026

## 1. Neural Acoustic Modeling for Tonewoods

### Key Paper: Neural Network-Based Tonewood Characterization (Badiane et al., 2023)

**Source**: Journal of the Acoustical Society of America, Vol. 154, Issue 2

**Key Findings**:
- Neural network method to estimate elastic properties of spruce thin plates
- Encodes information from both eigenfrequencies and eigenmodes
- Uses neural network to find optimal material parameters
- Speeds up finite element model updating by several orders of magnitude
- Applicable to characterizing tonewood acoustic properties

**Relevance to Starwood**:
- Can be used to create accurate tonewood embeddings from physical measurements
- Enables real-time parameter estimation for tonewood simulation
- Provides scientific basis for tonewood acoustic property modeling

---

## 2. Transfer Learning and Timbre Transfer

### Google Magenta Tone Transfer

**Source**: https://sites.research.google/tonetransfer

**Key Concept**:
- Collaborative project between Google and Magenta team
- Musicians' instrumental performances converted into ML models
- Enables transferring timbre from one instrument to another
- Uses DDSP (Differentiable Digital Signal Processing) architecture

**Application to Starwood**:
- Can transfer tonewood "character" from premium guitars to any input
- Real-time timbre transformation possible
- Pre-trained models can be fine-tuned for specific tonewoods

### Audio Style Transfer Techniques

**Key Papers**:
1. "Audio Style Transfer" (Grinstein et al., 2018) - 118 citations
2. "Deep Learning Methods for Music Style Transfer" (Cífka, 2021)
3. "FM Tone Transfer with Envelope Learning" (Caspe, 2023)

**Techniques**:
- Spectral analysis using STFT
- VGG-19 based feature extraction
- Latent space manipulation for timbre control
- Envelope learning for dynamic transfer

---

## 3. DDSP Architecture for Tonewood Synthesis

### Comprehensive Review: Hayes et al. (2024)

**Source**: Frontiers in Signal Processing - "A Review of Differentiable Digital Signal Processing for Music and Speech Synthesis" (70 citations)

**Key Components**:

1. **Encoder**: Maps input audio to control parameters
   - F0 (fundamental frequency)
   - Loudness
   - Latent timbre vector (z)

2. **Decoder/Synthesizer**: Generates audio from parameters
   - Harmonic oscillator bank
   - Filtered noise generator
   - Trainable reverb

3. **Loss Functions**:
   - Multi-scale spectral loss
   - Perceptual losses
   - Reconstruction loss

**DDSP Synthesis Pipeline**:
```
Input Audio → Encoder → [F0, Loudness, z] → Decoder → Synthesized Audio
                                ↓
                        Tonewood Conditioning
```

### DDSP for Polyphonic Guitar (Jonason et al., 2023-2024)

**Key Innovation**:
- String-wise MIDI-to-audio synthesis
- Four different system architectures compared
- Control features: F0, loudness, periodicity per string
- Unified architecture merging control synthesis sub-modules

---

## 4. Latent Space Control for Timbre

### Key Concepts:

1. **Latent Timbre Synthesis**:
   - Learn continuous latent space representing timbre variations
   - Interpolate between tonewoods in latent space
   - Guiding vector normalized [0,1] for intuitive control

2. **Variational Autoencoder (VAE) Approach**:
   - Encode tonewood characteristics into latent vectors
   - Decode to synthesis parameters
   - Enable smooth transitions between tonewood types

3. **Controllable Synthesis via MIDI-DDSP**:
   - Separate control of pitch, dynamics, and timbre
   - Real-time parameter manipulation
   - Expressive performance rendering

---

## 5. Physical Modeling Integration

### Differentiable Modal Synthesis (NeurIPS 2024)

**Key Approach**:
- Finite difference method for string simulation
- Physics-informed neural networks
- Combine physical accuracy with neural flexibility

### Application to Tonewood:
- Model wood resonance using physical equations
- Neural network learns correction factors
- Hybrid approach: physics + data-driven

---

## 6. Real-Time Implementation

### Key Technologies:

1. **RAVE (Realtime Audio Variational autoEncoder)**:
   - Real-time timbre transfer
   - Lightweight models for embedded deployment
   - Can make guitar sound like any target instrument

2. **Neural Amp Modeler**:
   - Open-source neural network guitar effects
   - Real-time inference on microcontrollers
   - Proven architecture for guitar processing

3. **Minifusion**:
   - Live timbre transferring system
   - Real-time instrument transformation
   - Plugin format for DAW integration

---

## 7. Proposed Improvements for Starwood Tonewood AI Model

### Architecture Enhancements:

1. **Multi-Scale Tonewood Encoder**:
   - Encode physical properties (density, Young's modulus, damping)
   - Encode acoustic properties (resonance, sustain, brightness)
   - Encode perceptual properties (warmth, clarity, complexity)

2. **Hierarchical Latent Space**:
   - Wood species level (Brazilian Rosewood vs Cocobolo)
   - Individual specimen level (grain pattern, age)
   - Playing context level (strumming vs fingerpicking)

3. **Physics-Informed DDSP**:
   - Integrate finite element model constraints
   - Differentiable resonance modeling
   - Body cavity simulation

4. **Transfer Learning Pipeline**:
   - Pre-train on large guitar audio dataset
   - Fine-tune on premium tonewood recordings
   - Few-shot adaptation for rare tonewoods

5. **Real-Time Optimization**:
   - Model quantization for embedded deployment
   - Streaming inference architecture
   - Latency < 10ms target

---

## 8. Data Requirements

### Training Data Needed:

1. **Premium Guitar Recordings**:
   - Brazilian Rosewood guitars (various luthiers)
   - Cocobolo guitars
   - Koa guitars
   - Madagascar Rosewood guitars
   - Various spruce tops (Adirondack, Sitka, Engelmann)

2. **Measurement Data**:
   - Impulse responses from guitar bodies
   - Frequency response measurements
   - Tap tone recordings
   - Physical property measurements (if available)

3. **Playing Styles**:
   - Fingerpicking
   - Strumming
   - Single note lines
   - Chord progressions
   - Various dynamics (pp to ff)

---

## References

1. Badiane et al. (2023). "A neural network-based method for spruce tonewood characterization." JASA.
2. Hayes et al. (2024). "A review of differentiable digital signal processing for music and speech synthesis." Frontiers in Signal Processing.
3. Jonason et al. (2024). "DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance." DAFx.
4. Engel et al. (2020). "DDSP: Differentiable Digital Signal Processing." ICLR.
5. Cífka (2021). "Deep learning methods for music style transfer." PhD Thesis.
6. Grinstein et al. (2018). "Audio style transfer." IEEE ICASSP.


---

## 9. Guitar Audio Datasets for Training

### GOAT Dataset (2025)

**Source**: arXiv:2509.22655v1 - "GOAT: A Large Dataset of Paired Guitar Audio Recordings and Tablatures"

**Dataset Specifications**:
- 5.9 hours of unique high-quality direct input (DI) audio recordings
- Electric guitars from various guitars and players
- 29.5 hours total with amplifier augmentation
- Fully annotated with Guitar Pro tablatures
- Includes string/fret numbers and expressive playing techniques

**Annotations Include**:
- Guitar Pro format files
- DadaGP tokens (text-like representation)
- MIDI versions for compatibility
- Playing techniques: bends, palm mutes, legatos, slides

**Data Augmentation Strategy**:
- Guitar amplifier rendering for timbral variety
- Multiple amp configurations and cabinet impulse responses
- Near-unlimited tonal variation possible

**Relevance to Starwood**:
- Can be used for training tonewood-conditioned synthesis models
- Tablature annotations enable tab-to-audio training
- Playing technique labels support articulation modeling
- DI recordings ideal for tonewood character application

### Other Guitar Datasets

| Dataset | Duration | Real Audio | MIDI | Tablature |
|---------|----------|------------|------|-----------|
| GuitarSet | 180 min | ✓ | ✓ | ✓ |
| EGDB | 60 min | ✓ | ✓ | ✗ |
| GAPS | 90 min | ✓ | ✓ | ✗ |
| IDMT-SMT-Guitar | 45 min | ✓ | ✓ | ✓ |
| Guitar-TECHS | 30 min | ✓ | ✓ | ✓ |
| SynthTab | Large | Synthetic | ✓ | ✓ |
| GOAT | 354 min | ✓ | ✓ | ✓ |

---

## 10. Transfer Learning Approaches

### One-to-Many Amplifier Modeling (2024)

**Key Innovation**: Tone embedding approach for zero-shot learning of unseen amps

**Application to Tonewood**:
- Create "tonewood embeddings" similar to amp tone embeddings
- Enable zero-shot transfer to new tonewood types
- Reference audio approach for capturing tonewood character

### Few-Shot Learning for Instruments

**Techniques**:
1. Meta-learning for rapid adaptation to new tonewoods
2. Prototypical networks for tonewood classification
3. Siamese networks for tonewood similarity matching

### Pre-trained Audio Models

**Available Models**:
- NSynth (Google Magenta) - instrument synthesis
- Jukebox (OpenAI) - music generation
- AudioLDM - text-to-audio
- MusicGen (Meta) - music generation

**Transfer Strategy for Starwood**:
1. Start with pre-trained audio encoder (e.g., from MusicGen)
2. Fine-tune on premium guitar recordings
3. Add tonewood conditioning layers
4. Train on tonewood-labeled dataset

---

## 11. Proposed Training Pipeline for Starwood Tonewood AI

### Phase 1: Data Collection
- Acquire recordings from premium guitars (Brazilian Rosewood, Cocobolo, etc.)
- Record same musical passages on different guitars
- Capture DI and mic'd recordings
- Document guitar specifications and tonewood details

### Phase 2: Pre-training
- Use GOAT dataset for general guitar synthesis pre-training
- Learn guitar articulations and playing techniques
- Establish baseline audio quality

### Phase 3: Tonewood Fine-tuning
- Fine-tune on tonewood-labeled recordings
- Learn tonewood embeddings
- Train tonewood conditioning modules

### Phase 4: Few-Shot Adaptation
- Enable rapid adaptation to new/rare tonewoods
- Use reference audio for zero-shot tonewood transfer
- Implement tonewood interpolation in latent space


---

## 12. Advanced Audio Synthesis Techniques

### Diffusion-Based Guitar Tone Morphing (2025)

**Source**: arXiv:2510.07908 - "Guitar Tone Morphing by Diffusion-based Model"

**Key Innovation**: Smooth transitions between guitar tones using latent diffusion models

**Four Methods Explored**:

1. **Without LoRA Fine-tuning**: Baseline using pre-trained LDM encoder, SLERP interpolation in latent space
2. **Single-Sided LoRA**: Fine-tune only conditional U-Net, better prompt control
3. **Dual-Sided LoRA**: Fine-tune separate U-Nets for each tone, interpolate parameters
4. **Music2Latent Interpolation**: Direct latent space interpolation without diffusion

**Key Techniques**:
- **SLERP (Spherical Linear Interpolation)**: Smooth interpolation on unit sphere for latent vectors
- **AdaIN (Adaptive Instance Normalization)**: Align style distributions between features
- **LoRA Fine-tuning**: Low-rank adaptation for efficient model customization

**Application to Starwood Tonewood**:
- Use SLERP to interpolate between tonewood latent embeddings
- Enable smooth morphing from "Brazilian Rosewood" to "Cocobolo" character
- Real-time tone blending for live performance
- Music2Latent approach ideal for lightweight deployment

### Neural Audio Codecs

**Key Technologies**:

1. **SoundStream (Google, 2021)**: End-to-end neural audio codec, 1241 citations
2. **EnCodec (Meta)**: High-fidelity audio compression for music
3. **DAC (Descript Audio Codec)**: High-quality audio tokenization
4. **MuCodec (2025)**: Ultra low-bitrate music codec for generation

**Relevance to Starwood**:
- Compress tonewood characteristics into discrete tokens
- Enable language model-based generation
- Efficient storage and transmission of tonewood profiles

### Latent Diffusion for Music

**Key Papers**:

1. **AudioLDM**: Text-to-audio using latent diffusion
2. **Stable Audio**: High-quality music generation
3. **Multi-Instrument Music Synthesis (ISMIR 2022)**: Spectrogram diffusion, 87 citations

**Architecture for Tonewood-Conditioned Generation**:
```
Text Prompt + Tonewood Embedding → Cross-Attention → U-Net → VAE Decoder → Audio
```

### Expressive Acoustic Guitar Synthesis (2024)

**Source**: arXiv:2401.13498 - Kim et al.

**Key Approach**:
- Diffusion-based outpainting for long-term consistency
- Instrument-agnostic synthesis framework
- Expressive performance rendering

**Application to Starwood**:
- Combine with tonewood conditioning
- Generate expressive performances with specific wood character
- Long-form coherent audio generation

---

## 13. Proposed Starwood Tonewood AI Architecture v2.0

Based on the research findings, here is the enhanced architecture:

### Core Components

1. **Tonewood Encoder**
   - Input: Physical properties + acoustic measurements + reference audio
   - Output: 512-dimensional tonewood embedding
   - Architecture: Transformer encoder with multi-modal fusion

2. **DDSP Synthesis Core** (existing)
   - Enhanced with tonewood conditioning at every layer
   - String-wise synthesis for polyphonic guitar
   - Articulation-aware processing

3. **Latent Diffusion Module** (NEW)
   - For high-fidelity audio generation
   - Tonewood-conditioned U-Net
   - SLERP interpolation for tonewood morphing

4. **Neural Audio Codec** (NEW)
   - Compress tonewood characteristics to tokens
   - Enable LLM-based music generation
   - Efficient storage of tonewood profiles

5. **Real-Time Inference Engine**
   - Optimized for < 10ms latency
   - LoRA adapters for quick tonewood switching
   - Streaming architecture

### Training Strategy

**Stage 1: Pre-training**
- Train on GOAT dataset (29.5 hours)
- Learn general guitar synthesis
- Establish baseline audio quality

**Stage 2: Tonewood Conditioning**
- Fine-tune on tonewood-labeled recordings
- Learn tonewood embeddings
- Train cross-attention modules

**Stage 3: Diffusion Enhancement**
- Add latent diffusion for high-fidelity output
- Train tonewood-conditioned U-Net
- Implement SLERP morphing

**Stage 4: Real-Time Optimization**
- Quantize models for deployment
- Implement streaming inference
- Create LoRA adapters for each tonewood

### Tonewood Morphing Pipeline

```
Input Audio → Encoder → Latent z₁
                         ↓
Tonewood A Embedding → SLERP(α) ← Tonewood B Embedding
                         ↓
              Morphed Latent z_morph
                         ↓
              Diffusion Decoder → Output Audio
```

This enables:
- Smooth transitions between any two tonewoods
- Real-time α control (0 = Tonewood A, 1 = Tonewood B)
- Creative blending of wood characteristics


---

## 14. Tonewood Characterization and Measurement Techniques

### Professional Tonewood Measurement Systems

#### TPC System (Tonewood Parameters Characterization)

**Source**: Iulius Guitars - https://www.iuliusguitars.com/tpc/

**Key Features**:
- Only commercial tool specifically designed for luthiers to measure tonewood damping
- Analyzes audio impulses from tap tests
- Calculates key mechanical properties:
  - **Q (Quality) Factor**: Measures resonance/damping
  - **Young's Modulus**: Stiffness measurement
  - **Density**: Mass per unit volume
  - **Radiation Coefficient**: Sound radiation efficiency

**Measurement Method**:
- Free beam method
- Sample excited on first longitudinal bending mode
- Excitation along wood grain
- Close-field microphone placement (5cm)
- USB measurement microphone (MiniDSP Umik-1 recommended)

**Data Export**:
- Complete measurement files
- Excel spreadsheet export
- Audio tap tone recordings
- Frequency response of complete instruments

**Price**: €900 (software + templates + samples)

#### Pacific Rim Tonewoods Sonic Grading System

**Source**: https://pacificrimtonewoods.com/pages/sonic-grading

**Research Background**:
- 5 years of research
- 6 labs on 2 continents
- 18 guitars built to exact specifications
- 58 trained listeners for evaluation
- Testing at Dresden Technical University anechoic chamber

**Key Parameters Measured**:
1. **Density**: Mass per unit volume
2. **Stiffness**: Resistance to deformation
3. **Damping (Q Factor)**: Energy dissipation rate

**Key Discovery**:
> "For the first time, we can accurately quantify the relevant acoustic properties of soundboard and bracing, and thus grade tonewood by criteria that will allow the optimal tonal quality."

**Grading Categories**:
- Low Q (more damping, warmer tone)
- Mid Q (balanced)
- High Q (less damping, brighter, more sustain)

### Tap Tone Analysis

**Traditional Luthier Method**:
- Tap wood with finger or mallet
- Listen for pitch, sustain, and overtones
- Subjective but effective with experience

**Modern Spectral Analysis**:
- Use FFT (Fast Fourier Transform) to analyze tap response
- Identify modal frequencies
- Measure decay rates
- Compare spectral signatures

**Key Frequencies to Analyze**:
- Fundamental mode (typically 100-300 Hz for soundboards)
- First overtone
- Cross-grain modes
- Higher harmonics

### Key Acoustic Properties for AI Training

| Property | Symbol | Unit | Typical Range (Spruce) | Significance |
|----------|--------|------|------------------------|--------------|
| Density | ρ | kg/m³ | 350-450 | Mass, affects volume |
| Young's Modulus (longitudinal) | E_L | GPa | 10-16 | Stiffness along grain |
| Young's Modulus (radial) | E_R | GPa | 0.5-1.2 | Stiffness across grain |
| Shear Modulus | G | GPa | 0.6-1.0 | Resistance to shear |
| Damping (tan δ) | Q⁻¹ | - | 0.006-0.012 | Energy loss per cycle |
| Sound Velocity | c | m/s | 4000-6000 | Speed of sound in wood |
| Radiation Ratio | R | m⁴/kg·s | 8-15 | Sound radiation efficiency |
| Specific Modulus | E/ρ | GPa·m³/kg | 25-40 | Stiffness-to-weight ratio |

### Data Collection Strategy for Starwood AI

**Phase 1: Physical Measurement Database**
1. Partner with tonewood suppliers (Pacific Rim, LMI, Allied Lutherie)
2. Measure 1000+ samples across all major species
3. Record: density, E_L, E_R, G, Q factor, radiation ratio
4. Store high-resolution tap tone recordings

**Phase 2: Audio Recording Database**
1. Record identical musical phrases on guitars with different tonewoods
2. Control for: strings, player, room, microphone
3. Capture: strumming, fingerpicking, single notes, chords
4. Include articulations: palm mute, harmonics, slides, bends

**Phase 3: Paired Dataset Creation**
1. Match physical properties to audio characteristics
2. Create tonewood embeddings from physical parameters
3. Train mapping from embeddings to audio features
4. Validate with blind listening tests

---

## 15. Enhanced Starwood Tonewood AI Architecture v3.0

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STARWOOD TONEWOOD AI v3.0                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Physical Props   │    │ Reference Audio  │                   │
│  │ (ρ, E, G, Q, R)  │    │ (Tap Tones)      │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌────────────────────────────────────────────┐                 │
│  │         TONEWOOD ENCODER (Transformer)      │                 │
│  │  - Multi-modal fusion                       │                 │
│  │  - Physical property embedding              │                 │
│  │  - Audio feature extraction                 │                 │
│  │  - Output: 512-dim tonewood embedding       │                 │
│  └────────────────────┬───────────────────────┘                 │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────┐                 │
│  │         SYNTHESIS CORE (Hybrid)             │                 │
│  │                                             │                 │
│  │  ┌─────────────┐  ┌─────────────┐          │                 │
│  │  │ DDSP Core   │  │ Diffusion   │          │                 │
│  │  │ (Real-time) │  │ (Hi-Fi)     │          │                 │
│  │  └──────┬──────┘  └──────┬──────┘          │                 │
│  │         │                │                  │                 │
│  │         ▼                ▼                  │                 │
│  │  ┌─────────────────────────────────┐       │                 │
│  │  │   Tonewood Conditioning Layer    │       │                 │
│  │  │   - Cross-attention with embed   │       │                 │
│  │  │   - FiLM modulation              │       │                 │
│  │  │   - AdaIN style transfer         │       │                 │
│  │  └─────────────────────────────────┘       │                 │
│  └────────────────────┬───────────────────────┘                 │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────┐                 │
│  │         TONEWOOD MORPHING ENGINE            │                 │
│  │  - SLERP interpolation between embeddings   │                 │
│  │  - Real-time α control (0.0 - 1.0)          │                 │
│  │  - Smooth transitions between woods         │                 │
│  └────────────────────┬───────────────────────┘                 │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────┐                 │
│  │         OUTPUT PROCESSING                   │                 │
│  │  - Neural vocoder (HiFi-GAN)               │                 │
│  │  - Room simulation                          │                 │
│  │  - Microphone modeling                      │                 │
│  └────────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Improvements Over v2.0

1. **Physical Property Integration**
   - Direct input of measured tonewood parameters
   - TPC-compatible data format
   - Automatic property estimation from audio

2. **Multi-Modal Tonewood Encoder**
   - Combines physical measurements + audio features
   - Transformer architecture for cross-modal attention
   - Produces unified 512-dim embedding

3. **Hybrid Synthesis Core**
   - DDSP for real-time (<10ms latency)
   - Diffusion for high-fidelity offline rendering
   - Automatic quality/latency tradeoff

4. **Advanced Tonewood Morphing**
   - SLERP interpolation in embedding space
   - Real-time morphing control
   - Perceptually smooth transitions

5. **Professional Measurement Integration**
   - Import from TPC system
   - Import from Pacific Rim Tonewoods data
   - Automatic calibration from reference recordings
