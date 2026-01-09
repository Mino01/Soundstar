# Starwood Tonewood AI Model Improvements

## Technical Specification for Enhanced Neural Tonewood Synthesis

**Version**: 3.0  
**Author**: Manus AI  
**Date**: January 2026  
**Repository**: https://github.com/Mino01/Starwood

---

## Executive Summary

This document presents a comprehensive technical specification for improving the Starwood Guitar Pro Sound Engine with advanced tonewood AI modeling techniques. The proposed enhancements leverage cutting-edge research in neural acoustic modeling, transfer learning from real guitar recordings, diffusion-based audio synthesis, and professional tonewood characterization methods.

The key innovations include a multi-modal tonewood encoder that combines physical measurements with audio features, a hybrid DDSP-Diffusion synthesis core for balancing real-time performance with high-fidelity output, and a tonewood morphing engine that enables smooth transitions between different wood characters using spherical linear interpolation (SLERP) in the learned embedding space.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Neural Acoustic Modeling for Tonewoods](#2-neural-acoustic-modeling-for-tonewoods)
3. [Transfer Learning from Real Guitar Recordings](#3-transfer-learning-from-real-guitar-recordings)
4. [Advanced Audio Synthesis Techniques](#4-advanced-audio-synthesis-techniques)
5. [Tonewood Characterization and Measurement](#5-tonewood-characterization-and-measurement)
6. [Enhanced Architecture v3.0](#6-enhanced-architecture-v30)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [References](#8-references)

---

## 1. Introduction

The Starwood Guitar Pro Sound Engine aims to deliver the tonal characteristics of premium tonewoods—such as Brazilian rosewood, Cocobolo, and aged Sitka spruce—through neural audio synthesis. This document outlines research findings and architectural improvements that will enable Starwood to produce more authentic, expressive, and controllable guitar sounds.

The improvements are organized into four research areas:

| Research Area | Key Technologies | Impact |
|---------------|------------------|--------|
| Neural Acoustic Modeling | CNN classification, spectral analysis | Accurate tonewood identification |
| Transfer Learning | GOAT dataset, paired recordings | Real-world tonal authenticity |
| Advanced Synthesis | Diffusion models, SLERP morphing | High-fidelity output, smooth transitions |
| Measurement Integration | TPC system, sonic grading | Professional-grade accuracy |

---

## 2. Neural Acoustic Modeling for Tonewoods

### 2.1 Tonewood Classification via Neural Networks

Recent research from the Journal of the Acoustical Society of America demonstrates that convolutional neural networks (CNNs) can accurately classify spruce tonewood quality from acoustic measurements [1]. The study achieved **96.3% accuracy** in distinguishing between different grades of spruce using spectral features extracted from tap-tone recordings.

The key insight is that tonewood quality correlates strongly with specific acoustic parameters:

| Parameter | Symbol | Significance | Measurement Method |
|-----------|--------|--------------|-------------------|
| Quality Factor | Q | Resonance/sustain | Decay rate analysis |
| Young's Modulus | E | Stiffness | Vibration frequency |
| Density | ρ | Mass distribution | Weight/volume ratio |
| Radiation Ratio | R | Sound projection | Velocity/impedance |

### 2.2 Differentiable Digital Signal Processing (DDSP)

Google Magenta's DDSP framework provides the foundation for Starwood's synthesis core [2]. The key innovation is making traditional signal processing operations differentiable, enabling end-to-end training with gradient descent.

For guitar synthesis, DDSP uses:

- **Harmonic oscillators**: Generate pitched content with controllable harmonics
- **Filtered noise**: Add realistic noise components (string buzz, fret noise)
- **Reverb module**: Trainable impulse response for room simulation
- **Tonewood conditioning**: Modulate all parameters based on wood embedding

### 2.3 Tone Transfer Technology

Google Magenta's Tone Transfer demonstrates that timbre can be separated from pitch and timing, enabling the "character" of one instrument to be applied to another [3]. This is directly applicable to tonewood transformation:

> "Tone Transfer uses machine learning to transform audio from one instrument to sound like another, while preserving the original melody and timing."

For Starwood, this means:
1. Extract the "tonewood character" from reference recordings
2. Apply that character to any input guitar signal
3. Preserve the player's performance nuances

---

## 3. Transfer Learning from Real Guitar Recordings

### 3.1 The GOAT Dataset

The Guitar-Oriented Audio Tablature (GOAT) dataset provides **29.5 hours** of paired guitar audio and tablature data [4]. This is invaluable for training Starwood's synthesis models because it includes:

- Synchronized audio and symbolic representations
- Multiple playing techniques and articulations
- Diverse musical styles and contexts
- High-quality studio recordings

### 3.2 Training Strategy

The recommended training approach uses a three-stage curriculum:

**Stage 1: Pre-training on GOAT**
- Learn general guitar synthesis from the large paired dataset
- Establish baseline audio quality and articulation handling
- Duration: ~100 GPU hours

**Stage 2: Tonewood Fine-tuning**
- Fine-tune on tonewood-labeled recordings
- Learn tonewood-specific embeddings
- Implement cross-attention conditioning
- Duration: ~50 GPU hours

**Stage 3: Diffusion Enhancement**
- Add latent diffusion for high-fidelity output
- Train tonewood-conditioned U-Net
- Implement SLERP morphing capability
- Duration: ~100 GPU hours

### 3.3 Data Augmentation

To expand the effective training set, Starwood should implement:

| Augmentation | Description | Benefit |
|--------------|-------------|---------|
| Pitch shifting | Transpose ±2 semitones | Generalization across keys |
| Time stretching | ±10% tempo variation | Robustness to timing |
| Room simulation | Convolution with IRs | Acoustic environment diversity |
| Noise injection | Low-level background noise | Real-world robustness |

---

## 4. Advanced Audio Synthesis Techniques

### 4.1 Diffusion-Based Guitar Tone Morphing

Recent research demonstrates that latent diffusion models can achieve smooth, perceptually natural transitions between guitar tones [5]. The study explores four methods:

1. **Without LoRA**: Baseline using pre-trained LDM encoder with SLERP interpolation
2. **Single-Sided LoRA**: Fine-tune only the conditional U-Net for better prompt control
3. **Dual-Sided LoRA**: Fine-tune separate U-Nets for each tone, interpolate parameters
4. **Music2Latent**: Direct latent space interpolation without diffusion

The **Music2Latent approach** is particularly relevant for Starwood because it offers:
- Simpler architecture than full diffusion
- Faster inference for real-time applications
- Effective SLERP interpolation in learned latent space

### 4.2 Spherical Linear Interpolation (SLERP)

SLERP provides geometrically consistent interpolation between tonewood embeddings on the unit sphere. Given two tonewood embeddings **v₀** (e.g., Brazilian Rosewood) and **v₁** (e.g., Cocobolo), SLERP computes:

```
SLERP(α, v₀, v₁) = sin((1-α)θ)/sin(θ) · v₀ + sin(αθ)/sin(θ) · v₁
```

Where θ is the angle between the normalized vectors and α ∈ [0, 1] controls the blend.

This enables:
- **Real-time morphing**: Smoothly transition between tonewoods during playback
- **Creative blending**: Create hybrid tonewoods (e.g., 70% Brazilian + 30% Cocobolo)
- **Perceptual smoothness**: Avoid artifacts from linear interpolation

### 4.3 Adaptive Instance Normalization (AdaIN)

AdaIN aligns the statistical properties (mean and variance) of one feature map to match another [6]. For tonewood synthesis:

```
AdaIN(x, y) = σ(y) · (x - μ(x))/σ(x) + μ(y)
```

This allows Starwood to:
- Transfer the "style" of one tonewood to another
- Maintain content (notes, timing) while changing character
- Enable fine-grained tonal control

### 4.4 Neural Audio Codecs

Modern neural audio codecs provide efficient representations for music generation:

| Codec | Organization | Bitrate | Quality | Relevance |
|-------|--------------|---------|---------|-----------|
| SoundStream | Google | 3-18 kbps | High | Foundation for audio tokenization |
| EnCodec | Meta | 1.5-24 kbps | Very High | Used in MusicGen |
| DAC | Descript | 8-16 kbps | Excellent | High-quality music compression |
| MuCodec | 2025 | Ultra-low | Good | Efficient for generation |

For Starwood, neural codecs enable:
- Compact tonewood profile storage
- Language model-based generation
- Efficient streaming and transmission

---

## 5. Tonewood Characterization and Measurement

### 5.1 Professional Measurement Systems

#### TPC System (Tonewood Parameters Characterization)

The TPC system from Iulius Guitars is the only commercial tool specifically designed for luthiers to measure tonewood properties [7]. It calculates:

- **Q (Quality) Factor**: Measures resonance and damping
- **Young's Modulus**: Quantifies stiffness
- **Density**: Mass per unit volume
- **Radiation Coefficient**: Sound radiation efficiency

The measurement uses the **free beam method**, exciting the sample on its first longitudinal bending mode along the wood grain.

#### Pacific Rim Tonewoods Sonic Grading

Pacific Rim Tonewoods conducted a landmark 5-year study involving 6 laboratories, 18 identically-built guitars, and 58 trained listeners [8]. Their key finding:

> "For the first time, we can accurately quantify the relevant acoustic properties of soundboard and bracing, and thus grade tonewood by criteria that will allow the optimal tonal quality."

The sonic grading system categorizes tonewoods by:
- **Low Q**: More damping, warmer tone, faster decay
- **Mid Q**: Balanced response
- **High Q**: Less damping, brighter tone, longer sustain

### 5.2 Key Acoustic Properties for AI Training

The following properties should be captured for each tonewood sample in Starwood's training database:

| Property | Symbol | Unit | Typical Range (Spruce) | AI Feature |
|----------|--------|------|------------------------|------------|
| Density | ρ | kg/m³ | 350-450 | Direct input |
| Young's Modulus (L) | E_L | GPa | 10-16 | Direct input |
| Young's Modulus (R) | E_R | GPa | 0.5-1.2 | Direct input |
| Shear Modulus | G | GPa | 0.6-1.0 | Direct input |
| Damping | tan δ | - | 0.006-0.012 | Direct input |
| Sound Velocity | c | m/s | 4000-6000 | Derived |
| Radiation Ratio | R | m⁴/kg·s | 8-15 | Derived |
| Specific Modulus | E/ρ | GPa·m³/kg | 25-40 | Derived |

### 5.3 Data Collection Strategy

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

## 6. Enhanced Architecture v3.0

### 6.1 System Overview

The Starwood Tonewood AI v3.0 architecture integrates all research findings into a unified system:

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

### 6.2 Component Specifications

#### Tonewood Encoder

| Component | Specification |
|-----------|---------------|
| Architecture | Transformer encoder with 6 layers |
| Input (Physical) | 8-dimensional property vector |
| Input (Audio) | 128-dim mel-spectrogram features |
| Fusion | Cross-attention between modalities |
| Output | 512-dimensional tonewood embedding |
| Training | Contrastive learning on paired data |

#### Synthesis Core

| Mode | Latency | Quality | Use Case |
|------|---------|---------|----------|
| DDSP (Real-time) | < 10ms | Good | Live performance, hardware |
| Diffusion (Hi-Fi) | 500ms-2s | Excellent | Studio rendering, export |
| Hybrid | Adaptive | Balanced | Automatic quality/latency |

#### Tonewood Morphing Engine

| Feature | Specification |
|---------|---------------|
| Interpolation | SLERP on unit sphere |
| Control | Real-time α parameter (0.0-1.0) |
| Presets | 8+ premium tonewoods |
| Custom | User-defined blends |
| Automation | DAW parameter automation |

### 6.3 Key Improvements Over v2.0

1. **Physical Property Integration**: Direct input of measured tonewood parameters from TPC system or Pacific Rim Tonewoods data

2. **Multi-Modal Tonewood Encoder**: Combines physical measurements with audio features using transformer cross-attention

3. **Hybrid Synthesis Core**: DDSP for real-time applications, diffusion for high-fidelity offline rendering

4. **Advanced Tonewood Morphing**: SLERP interpolation enables perceptually smooth transitions between any two tonewoods

5. **Professional Measurement Compatibility**: Import data from industry-standard measurement systems

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Months 1-3)

| Task | Duration | Deliverable |
|------|----------|-------------|
| Dataset collection | 6 weeks | 1000+ tonewood samples |
| GOAT integration | 2 weeks | Pre-training pipeline |
| DDSP baseline | 4 weeks | Real-time synthesis v1 |

### Phase 2: Tonewood Conditioning (Months 4-6)

| Task | Duration | Deliverable |
|------|----------|-------------|
| Tonewood encoder | 4 weeks | 512-dim embeddings |
| Conditioning layers | 4 weeks | FiLM + cross-attention |
| Fine-tuning | 4 weeks | Tonewood-aware synthesis |

### Phase 3: Advanced Features (Months 7-9)

| Task | Duration | Deliverable |
|------|----------|-------------|
| Diffusion module | 6 weeks | Hi-Fi rendering |
| SLERP morphing | 3 weeks | Real-time blending |
| AdaIN transfer | 3 weeks | Style transfer capability |

### Phase 4: Integration & Polish (Months 10-12)

| Task | Duration | Deliverable |
|------|----------|-------------|
| Guitar Pro integration | 4 weeks | Tab-to-audio pipeline |
| VST plugin | 4 weeks | DAW integration |
| Hardware optimization | 4 weeks | Embedded deployment |

---

## 8. References

[1] Journal of the Acoustical Society of America. "A neural network-based method for spruce tonewood characterization." JASA 154(2), 730, 2023. https://pubs.aip.org/asa/jasa/article/154/2/730/2906397

[2] Google Magenta. "DDSP: Differentiable Digital Signal Processing." https://magenta.tensorflow.org/ddsp

[3] Google Magenta. "Tone Transfer." https://sites.research.google/tonetransfer

[4] arXiv. "GOAT: Guitar-Oriented Audio Tablature Dataset." arXiv:2509.22655, 2025. https://arxiv.org/html/2509.22655v1

[5] arXiv. "Guitar Tone Morphing by Diffusion-based Model." arXiv:2510.07908, 2025. https://arxiv.org/html/2510.07908

[6] Huang, X. and Belongie, S. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization." ICCV 2017.

[7] Iulius Guitars. "TPC System - Tonewood Parameters Characterization." https://www.iuliusguitars.com/tpc/

[8] Pacific Rim Tonewoods. "Sonic Grading." https://pacificrimtonewoods.com/pages/sonic-grading

[9] Frontiers in Signal Processing. "Differentiable Digital Signal Processing for Music Synthesis." 2023. https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1284100/full

[10] Springer. "Properties and Treatment of Tonewood for String Instruments." 2025. https://link.springer.com/chapter/10.1007/978-3-032-00824-4_4

---

## Appendix A: Tonewood Property Database Schema

```sql
CREATE TABLE tonewoods (
    id SERIAL PRIMARY KEY,
    species VARCHAR(100) NOT NULL,
    common_name VARCHAR(100),
    origin VARCHAR(100),
    
    -- Physical Properties
    density_kg_m3 FLOAT,
    youngs_modulus_longitudinal_gpa FLOAT,
    youngs_modulus_radial_gpa FLOAT,
    shear_modulus_gpa FLOAT,
    damping_tan_delta FLOAT,
    
    -- Derived Properties
    sound_velocity_m_s FLOAT,
    radiation_ratio FLOAT,
    specific_modulus FLOAT,
    
    -- Sonic Grading
    q_factor FLOAT,
    sonic_grade VARCHAR(20), -- 'low', 'mid', 'high'
    
    -- Audio Reference
    tap_tone_audio_path VARCHAR(255),
    reference_recording_path VARCHAR(255),
    
    -- Embedding
    embedding_512 FLOAT[512],
    
    -- Metadata
    measurement_date TIMESTAMP,
    measurement_system VARCHAR(50), -- 'TPC', 'PRT', 'custom'
    notes TEXT
);
```

---

## Appendix B: API Endpoints

### Tonewood Embedding

```
POST /api/v1/tonewood/embed
```

**Request:**
```json
{
    "physical_properties": {
        "density": 420.0,
        "youngs_modulus_l": 12.5,
        "youngs_modulus_r": 0.8,
        "shear_modulus": 0.7,
        "damping": 0.008
    },
    "tap_tone_audio": "base64_encoded_audio"
}
```

**Response:**
```json
{
    "embedding": [0.123, -0.456, ...],  // 512 floats
    "predicted_species": "Sitka Spruce",
    "sonic_grade": "high",
    "confidence": 0.94
}
```

### Tonewood Morphing

```
POST /api/v1/tonewood/morph
```

**Request:**
```json
{
    "tonewood_a": "brazilian_rosewood",
    "tonewood_b": "cocobolo",
    "alpha": 0.3,
    "input_audio": "base64_encoded_audio"
}
```

**Response:**
```json
{
    "output_audio": "base64_encoded_audio",
    "blend_description": "70% Brazilian Rosewood + 30% Cocobolo",
    "processing_time_ms": 450
}
```

---

*Document generated by Manus AI for the Starwood project.*
