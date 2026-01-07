# Boutique Guitar Sound Synthesis Technical Specification

**Project**: Starwood  
**Version**: 1.0  
**Date**: January 2026  
**Author**: Manus AI

---

## Executive Summary

This technical specification documents the research findings and implementation guidelines for synthesizing **boutique guitar sounds** within the Starwood framework. The specification covers three primary domains: **tonewood acoustic properties**, **microphone recording techniques**, and **AI/ML-based sound modeling**. By combining these elements, Starwood can generate authentic, high-fidelity acoustic guitar sounds that capture the unique tonal characteristics of premium tonewoods such as Brazilian rosewood and Cocobolo.

The proposed system leverages the **RAVE-DDSP-Transformer Tri-Hybrid Architecture** already established in Starwood, extending it with a specialized **Tonewood Neural Synthesis Engine** that models the physical and acoustic properties of different wood combinations.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Tonewood Acoustic Properties](#2-tonewood-acoustic-properties)
3. [Premium Tonewood Profiles](#3-premium-tonewood-profiles)
4. [Microphone Recording Techniques](#4-microphone-recording-techniques)
5. [AI/ML Sound Modeling Architecture](#5-aiml-sound-modeling-architecture)
6. [Implementation Guidelines](#6-implementation-guidelines)
7. [API Design](#7-api-design)
8. [Data Models](#8-data-models)
9. [References](#9-references)

---

## 1. Introduction

### 1.1 Background

The acoustic guitar produces its distinctive sound through a complex interaction of vibrating strings, resonating body panels, and the air cavity within the instrument. The choice of **tonewood**—the wood used for the guitar's top, back, and sides—has a profound impact on the instrument's tonal characteristics. Premium tonewoods such as Brazilian rosewood and Cocobolo are prized for their exceptional acoustic properties, producing sounds described as rich, complex, and full of overtones [1].

### 1.2 Objectives

This specification aims to:

1. Document the acoustic properties of premium tonewoods and their impact on guitar sound
2. Establish microphone placement techniques for capturing authentic guitar tones
3. Design an AI/ML architecture for synthesizing tonewood-specific sounds
4. Provide implementation guidelines for integration with Starwood's existing engine

### 1.3 Scope

The specification covers:

- **Back and side tonewoods**: Brazilian rosewood, Cocobolo, East Indian rosewood, Honduran mahogany, Koa
- **Top tonewoods**: Adirondack spruce, Sitka spruce, European spruce, Western red cedar
- **Microphone techniques**: Single-mic, stereo (X/Y, spaced pair, M/S, ORTF)
- **AI/ML approaches**: Neural tonewood characterization, DDSP synthesis, diffusion-based generation

---

## 2. Tonewood Acoustic Properties

### 2.1 Fundamental Principles

The acoustic properties of wood are primarily determined by its **elastic properties**, which govern how the material vibrates in response to string excitation [2]. The key parameters include:

| Parameter | Symbol | Description | Unit |
|-----------|--------|-------------|------|
| **Longitudinal Young's Modulus** | E_L | Stiffness along the grain | GPa |
| **Radial Young's Modulus** | E_R | Stiffness across the grain | GPa |
| **Shear Modulus** | G_LR | Resistance to shearing forces | GPa |
| **Density** | ρ | Mass per unit volume | kg/m³ |
| **Sound Velocity** | c | Speed of sound through wood | m/s |
| **Damping Coefficient** | tan δ | Internal friction (affects sustain) | - |

The **sound velocity** is derived from the relationship:

> c = √(E_L / ρ)

Higher sound velocity generally correlates with brighter, more responsive tones, while lower damping coefficients result in longer sustain and richer overtones [3].

### 2.2 Resonance Mechanisms

An acoustic guitar produces sound through three primary resonance mechanisms:

1. **Panel Resonances**: The wooden top (soundboard) and back vibrate in complex patterns, each contributing different frequency components to the overall sound.

2. **Air Resonance (Helmholtz Resonance)**: The body of air contained within the guitar resonates at a frequency typically around 100Hz, providing low-end projection and sustain [4].

3. **String Resonance**: The vibrating strings produce the fundamental pitch and harmonics, which are then amplified and colored by the body resonances.

### 2.3 Tonewood Classification

Tonewoods are broadly classified by their acoustic function:

| Category | Function | Preferred Properties |
|----------|----------|---------------------|
| **Soundboard (Top)** | Primary sound radiation | High stiffness-to-weight ratio, low damping |
| **Back/Sides** | Reflection and coloring | High density, complex grain, low damping |
| **Neck** | Structural support | Stability, moderate stiffness |
| **Bracing** | Structural reinforcement | High stiffness, low mass |

---

## 3. Premium Tonewood Profiles

### 3.1 Brazilian Rosewood (Dalbergia nigra)

Brazilian rosewood is widely regarded as the **"Holy Grail"** of tonewoods, prized for its exceptional acoustic properties and visual beauty [5].

#### Physical Properties

| Property | Value | Notes |
|----------|-------|-------|
| **Scientific Name** | Dalbergia nigra | CITES Appendix I (highest protection) |
| **Density** | 850-950 kg/m³ | Very high |
| **E_L** | 13.5 GPa | High stiffness |
| **E_R** | 1.8 GPa | Good radial stiffness |
| **G_LR** | 1.2 GPa | Excellent shear resistance |
| **Sound Velocity** | ~4000 m/s | Fast response |
| **Damping** | Very low | Exceptional sustain |

#### Tonal Characteristics

> "There is a genuine, tangible weight to the bass, which sweeps and swells underneath every aspect of the instrument's tone, anchoring your music and embracing it at the same time. No other tonewood provides such a blossom—warm, strong and supportive." — Bedell Guitars [5]

Brazilian rosewood is characterized by:

- **Bass**: Round, rumbling, with exceptional depth and weight
- **Midrange**: Warm, strong, supportive presence
- **Trebles**: Sparkling, shimmering, with "glass-like ring"
- **Overtones**: Rich, complex, reverb-laden cornucopia
- **Sustain**: Mind-boggling, eternal
- **Dynamic Range**: Exceptional—responds to whisper-soft fingerpicking and aggressive flatpicking equally well

#### Optimal Pairings

Brazilian rosewood pairs exceptionally well with **Adirondack spruce** tops, creating the classic "pre-war" sound favored by bluegrass and fingerstyle players.

### 3.2 Cocobolo (Dalbergia retusa)

Cocobolo is one of the densest and most visually striking tonewoods, offering a tone that rivals Brazilian rosewood [6].

#### Physical Properties

| Property | Value | Notes |
|----------|-------|-------|
| **Scientific Name** | Dalbergia retusa | CITES Appendix II |
| **Density** | 1000-1200 kg/m³ | Negatively buoyant (sinks in water) |
| **E_L** | 15.0 GPa | Very high stiffness |
| **E_R** | 2.0 GPa | Excellent radial stiffness |
| **G_LR** | 1.4 GPa | Superior shear resistance |
| **Sound Velocity** | ~3800 m/s | Fast response |
| **Damping** | Extremely low | Outstanding sustain |
| **Oil Content** | Very high | Requires special finishing |

#### Tonal Characteristics

> "A dense, stiff tropical hardwood from Mexico, cocobolo produces a fairly bright overall tone emphasized by sparkling treble notes. Sonically it resembles koa but resonates a little deeper on the low end." — Taylor Guitars [6]

Cocobolo is characterized by:

- **Bass**: Deep, resonant, similar to Brazilian rosewood
- **Midrange**: Strong presence, well-defined
- **Trebles**: Sparkling, zingy, with excellent articulation
- **Response**: Fast, responsive, articulate
- **Sustain**: Excellent with moderate note decay
- **Overall**: Brighter than Brazilian, with more treble emphasis

#### Comparison with Brazilian Rosewood

| Aspect | Brazilian Rosewood | Cocobolo |
|--------|-------------------|----------|
| **Density** | 850-950 kg/m³ | 1000-1200 kg/m³ |
| **Bass** | Warmer, more enveloping | Deep but slightly tighter |
| **Trebles** | Shimmering | Sparkling, more pronounced |
| **Overtones** | More complex | Rich but slightly simpler |
| **Sustain** | Longer | Excellent but slightly shorter |
| **Brightness** | Balanced | Brighter overall |

### 3.3 Additional Premium Tonewoods

#### East Indian Rosewood (Dalbergia latifolia)

The most common rosewood substitute, offering warm, balanced tone with good bass response. Less complex overtones than Brazilian but more affordable and sustainable.

#### Honduran Mahogany (Swietenia macrophylla)

A medium-density hardwood producing warm, woody tones with strong midrange presence. Favored for folk and singer-songwriter applications.

#### Hawaiian Koa (Acacia koa)

A visually stunning wood with bright, focused tone similar to mahogany but with more high-end sparkle. Tone "opens up" with playing over time.

### 3.4 Soundboard Tonewoods

#### Adirondack Spruce (Picea rubens)

The premium choice for acoustic guitar tops, offering exceptional stiffness-to-weight ratio and dynamic range. Produces powerful, articulate tone that "opens up" with playing.

| Property | Value |
|----------|-------|
| **Density** | 400-450 kg/m³ |
| **E_L** | 12.5 GPa |
| **Sound Velocity** | ~5500 m/s |
| **Character** | Powerful, dynamic, articulate |

#### Sitka Spruce (Picea sitchensis)

The most common guitar top wood, offering balanced tone suitable for a wide range of playing styles.

#### European Spruce (Picea abies)

Traditional choice for classical guitars, offering warm, complex tone with excellent projection.

---

## 4. Microphone Recording Techniques

### 4.1 Understanding Guitar Sound Dispersion

The acoustic guitar is not a point source—it radiates different frequencies in different directions from various parts of the instrument [4]. Understanding this dispersion is critical for capturing authentic tones.

#### Key Radiation Zones

| Zone | Frequency Content | Character |
|------|------------------|-----------|
| **Soundhole** | Low frequencies (air resonance ~100Hz) | Boomy, powerful bass |
| **Bridge Area** | Mid-low frequencies (panel resonances) | Warm, full body |
| **12th Fret** | Mid-high frequencies (string sound) | Bright, articulate |
| **Upper Bout** | High frequencies | Airy, detailed |

### 4.2 Single Microphone Positions

#### Position 1: "Vanilla Position" (12th Fret)

The most commonly recommended position, pointing at the junction between neck and body.

```
Distance: 6-12 inches
Angle: 30-45° toward headstock
Character: Balanced body resonance + string liveliness
Best For: Busy mixes, strumming, general purpose
```

#### Position 2: Front Body Position (Albini/Schmitt Method)

Positioned in front of the guitar body, horizontally aligned with soundhole but vertically offset.

```
Distance: 12-18 inches
Angle: Looking down at guitar (above soundhole)
Character: Full body resonance, natural projection
Best For: Solo acoustic, fingerstyle, classical
```

#### Position 3: Bridge Arc Position

Positioned in an arc around the bridge, pointing toward it.

```
Distance: 8-14 inches
Angle: Pointing at bridge
Character: Warm, less string emphasis
Best For: Fingerpicking, reducing fret noise
```

### 4.3 Stereo Microphone Techniques

#### X/Y (Coincident Pair)

Two small diaphragm condensers with capsules nearly touching, angled 90-120° apart.

| Aspect | Specification |
|--------|--------------|
| **Mic Spacing** | Capsules touching |
| **Angle** | 90-120° apart |
| **Mono Compatibility** | Excellent |
| **Stereo Width** | Moderate |
| **Phase Issues** | None |

#### Spaced Pair (A/B)

Two mics separated by 12-24 inches, typically one at 12th fret and one at bridge.

| Aspect | Specification |
|--------|--------------|
| **Mic Spacing** | 12-24 inches |
| **Angle** | Parallel or slight toe-in |
| **Mono Compatibility** | Fair (check phase) |
| **Stereo Width** | Wide |
| **Phase Issues** | Possible |

#### Mid-Side (M/S)

Cardioid mic (mid) combined with figure-8 mic (side), decoded in post-production.

| Aspect | Specification |
|--------|--------------|
| **Mid Mic** | Cardioid, pointing at source |
| **Side Mic** | Figure-8, perpendicular to source |
| **Mono Compatibility** | Perfect |
| **Stereo Width** | Adjustable in post |
| **Flexibility** | Maximum |

#### ORTF

Two cardioid mics, 17cm apart, angled 110° apart (French broadcasting standard).

| Aspect | Specification |
|--------|--------------|
| **Mic Spacing** | 17cm (6.7 inches) |
| **Angle** | 110° apart |
| **Mono Compatibility** | Good |
| **Stereo Width** | Natural |
| **Character** | Realistic stereo image |

### 4.4 Recommended Microphones

#### Small Diaphragm Condensers (SDC)

| Microphone | Character | Frequency Response | Price Tier |
|------------|-----------|-------------------|------------|
| **Neumann KM 184** | Transparent, detailed, low noise | 20Hz-20kHz (slight HF lift) | Premium |
| **AKG C451 B** | Classic, bright, articulate | 20Hz-20kHz (presence peak) | Professional |
| **Shure SM81** | Neutral, reliable | 20Hz-20kHz (flat) | Professional |
| **Audio-Technica AT4041** | Balanced, versatile | 20Hz-20kHz | Mid-range |

#### Large Diaphragm Condensers (LDC)

| Microphone | Character | Frequency Response | Price Tier |
|------------|-----------|-------------------|------------|
| **Neumann U87** | Rich, full, industry standard | 20Hz-20kHz (slight presence) | Premium |
| **AKG C414** | Versatile, multiple patterns | 20Hz-20kHz | Professional |
| **Audio-Technica AT4050** | Warm, detailed | 20Hz-20kHz | Professional |

### 4.5 Microphone Modeling Parameters

For Starwood to simulate different microphone setups, the following parameters must be modeled:

```python
class MicrophoneModel:
    frequency_response: List[Tuple[float, float]]  # (frequency_hz, gain_db)
    polar_pattern: str  # "cardioid", "omni", "figure8", "hypercardioid"
    proximity_effect: float  # 0.0-1.0 (bass boost at close distances)
    self_noise: float  # dB SPL equivalent
    max_spl: float  # dB SPL before distortion
    transient_response: float  # ms (attack time)
```

---

## 5. AI/ML Sound Modeling Architecture

### 5.1 Overview

The proposed **Tonewood Neural Synthesis Engine** extends Starwood's RAVE-DDSP-Transformer architecture with specialized modules for modeling tonewood acoustic properties and microphone characteristics.

### 5.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TONEWOOD NEURAL SYNTHESIS ENGINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  TONEWOOD        │    │  GUITAR BODY     │    │  MICROPHONE      │  │
│  │  EMBEDDING       │───▶│  RESONANCE       │───▶│  SIMULATION      │  │
│  │  NETWORK         │    │  MODEL           │    │  MODULE          │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                       │                       │             │
│           ▼                       ▼                       ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    DDSP SYNTHESIS CORE                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  │  │ Harmonic   │  │ Filtered   │  │ Modal      │  │ Reverb     │  │  │
│  │  │ Oscillator │  │ Noise      │  │ Resonator  │  │ Module     │  │  │
│  │  │ Bank       │  │ Generator  │  │ Network    │  │            │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│                    ┌──────────────────┐                                │
│                    │  OUTPUT AUDIO    │                                │
│                    │  (48kHz/24-bit)  │                                │
│                    └──────────────────┘                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Component Details

#### 5.3.1 Tonewood Embedding Network

The Tonewood Embedding Network learns a dense vector representation of tonewood acoustic properties.

**Input Features**:
- Density (kg/m³)
- Longitudinal Young's Modulus (GPa)
- Radial Young's Modulus (GPa)
- Shear Modulus (GPa)
- Damping Coefficient
- Grain Pattern Features (extracted from images)

**Architecture**:
```python
class TonewoodEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=10, embedding_dim=256):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, tonewood_features):
        return self.encoder(tonewood_features)
```

**Output**: 256-dimensional embedding vector capturing tonewood tonal characteristics.

#### 5.3.2 Guitar Body Resonance Model

The Guitar Body Resonance Model predicts the frequency response of a guitar body based on tonewood embeddings and body geometry.

**Inputs**:
- Top tonewood embedding (256-dim)
- Back/sides tonewood embedding (256-dim)
- Body shape parameters (dreadnought, OM, concert, etc.)
- Bracing pattern encoding
- Scale length

**Architecture**:
- Transformer encoder for multi-tonewood interaction
- MLP decoder for frequency response prediction
- Modal decomposition for resonance peaks

**Outputs**:
- Frequency response curve (20Hz-20kHz)
- Modal frequencies and Q-factors
- Decay characteristics per frequency band

#### 5.3.3 Microphone Simulation Module

The Microphone Simulation Module applies realistic microphone coloration to the synthesized signal.

**Parameters**:
- Microphone model (U87, KM184, C414, etc.)
- Position (distance, angle, height)
- Polar pattern
- Room characteristics

**Processing**:
1. Apply position-dependent frequency filtering
2. Model proximity effect (bass boost at close distances)
3. Apply polar pattern attenuation
4. Add realistic self-noise

#### 5.3.4 DDSP Synthesis Core

The DDSP Synthesis Core generates the final audio waveform using differentiable signal processing [7].

**Components**:

1. **Harmonic Oscillator Bank**: Generates the tonal components (fundamental + harmonics)
2. **Filtered Noise Generator**: Adds transient and textural components (pick attack, string buzz)
3. **Modal Resonator Network**: Applies body resonance characteristics
4. **Differentiable Reverb**: Simulates room acoustics

### 5.4 Training Strategy

#### Dataset Requirements

| Data Type | Source | Quantity |
|-----------|--------|----------|
| **Tonewood Measurements** | Luthier workshops, research labs | 500+ samples |
| **Guitar Recordings** | Professional studios, sample libraries | 10,000+ hours |
| **Impulse Responses** | Guitar body IRs, room IRs | 1,000+ IRs |
| **MIDI/Audio Pairs** | Performance datasets, synthesized data | 50,000+ pairs |

#### Training Phases

1. **Phase 1**: Pre-train tonewood embedding on acoustic measurement data
2. **Phase 2**: Train body resonance model on guitar impulse responses
3. **Phase 3**: Fine-tune DDSP synthesis on high-quality recordings
4. **Phase 4**: End-to-end training with perceptual loss functions

#### Loss Functions

```python
total_loss = (
    spectral_loss +           # Multi-resolution STFT loss
    perceptual_loss +         # VGGish perceptual features
    tonewood_consistency +    # Embedding consistency across samples
    physical_plausibility     # Physics-informed constraints
)
```

---

## 6. Implementation Guidelines

### 6.1 Integration with Starwood Architecture

The Tonewood Neural Synthesis Engine integrates with Starwood's existing RAVE-DDSP-Transformer architecture as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STARWOOD TRI-HYBRID ENGINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │   RAVE      │   │ STRUCTURAL  │   │  TONEWOOD NEURAL    │   │
│  │   LATENT    │◀──│ TRANSFORMER │◀──│  SYNTHESIS ENGINE   │   │
│  │   SPACE     │   │             │   │  (NEW)              │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│         │                                      │                │
│         ▼                                      │                │
│  ┌─────────────┐                              │                │
│  │   DDSP      │◀─────────────────────────────┘                │
│  │ REFINEMENT  │                                               │
│  │   LAYER     │                                               │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │  ULTIMATE   │                                               │
│  │   MIXER     │                                               │
│  └─────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Natural Language Control via Aurora LLM

Users can control tonewood synthesis through natural language prompts processed by Aurora LLM:

**Example Prompts**:

| User Prompt | Parsed Parameters |
|-------------|------------------|
| "Brazilian rosewood with Adirondack top, warm fingerpicking tone" | `tonewood_back="brazilian_rosewood", tonewood_top="adirondack_spruce", style="fingerpicking", warmth=0.7` |
| "Bright cocobolo sound, close-miked with a U87" | `tonewood_back="cocobolo", mic_model="neumann_u87", mic_distance=0.2, brightness=0.8` |
| "Vintage Martin D-28 tone, room ambience" | `preset="martin_d28_vintage", reverb=0.4, room_size="medium"` |

### 6.3 Real-Time Performance Considerations

For real-time synthesis, the following optimizations are recommended:

| Component | Optimization | Latency Target |
|-----------|-------------|----------------|
| **Tonewood Embedding** | Pre-compute and cache | < 1ms |
| **Body Resonance** | Use lookup tables for common configurations | < 5ms |
| **DDSP Synthesis** | GPU acceleration, streaming processing | < 10ms |
| **Microphone Simulation** | FIR filter convolution | < 2ms |
| **Total Pipeline** | | < 20ms |

---

## 7. API Design

### 7.1 Tonewood Configuration Endpoint

```
POST /api/v1/starwood/tonewood/configure
```

**Request Body**:
```json
{
  "guitar_config": {
    "body_shape": "dreadnought",
    "scale_length": 25.4,
    "bracing_pattern": "x_bracing"
  },
  "tonewood_config": {
    "top": {
      "species": "adirondack_spruce",
      "grade": "master",
      "custom_properties": null
    },
    "back_sides": {
      "species": "brazilian_rosewood",
      "grade": "premium",
      "custom_properties": null
    }
  },
  "microphone_config": {
    "model": "neumann_km184",
    "position": {
      "distance_inches": 12,
      "angle_degrees": 30,
      "target": "12th_fret"
    },
    "stereo_technique": "xy"
  }
}
```

**Response**:
```json
{
  "config_id": "cfg_abc123",
  "tonewood_embedding": [0.123, -0.456, ...],
  "predicted_frequency_response": {
    "frequencies_hz": [20, 50, 100, ...],
    "magnitude_db": [-3.2, 0.5, 2.1, ...]
  },
  "estimated_characteristics": {
    "bass_weight": 0.85,
    "midrange_presence": 0.72,
    "treble_sparkle": 0.78,
    "sustain": 0.92,
    "overtone_complexity": 0.88
  }
}
```

### 7.2 Synthesis Endpoint

```
POST /api/v1/starwood/tonewood/synthesize
```

**Request Body**:
```json
{
  "config_id": "cfg_abc123",
  "input": {
    "type": "midi",
    "data": "base64_encoded_midi_data"
  },
  "output_format": {
    "sample_rate": 48000,
    "bit_depth": 24,
    "channels": "stereo"
  }
}
```

**Response**:
```json
{
  "job_id": "job_xyz789",
  "status": "processing",
  "estimated_duration_ms": 2500
}
```

### 7.3 Preset Retrieval Endpoint

```
GET /api/v1/starwood/tonewood/presets
```

**Response**:
```json
{
  "presets": [
    {
      "id": "preset_martin_d28",
      "name": "Martin D-28 (1940s)",
      "description": "Classic pre-war dreadnought tone",
      "tonewood_config": {
        "top": "adirondack_spruce",
        "back_sides": "brazilian_rosewood"
      }
    },
    {
      "id": "preset_taylor_814ce",
      "name": "Taylor 814ce",
      "description": "Modern grand auditorium with Indian rosewood",
      "tonewood_config": {
        "top": "sitka_spruce",
        "back_sides": "east_indian_rosewood"
      }
    }
  ]
}
```

---

## 8. Data Models

### 8.1 Database Schema

```sql
-- Tonewood species definitions
CREATE TABLE tonewood_species (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    scientific_name VARCHAR(150),
    cites_status ENUM('appendix_i', 'appendix_ii', 'appendix_iii', 'none'),
    typical_use ENUM('top', 'back_sides', 'neck', 'bracing'),
    density_min FLOAT,
    density_max FLOAT,
    youngs_modulus_l FLOAT,
    youngs_modulus_r FLOAT,
    shear_modulus FLOAT,
    sound_velocity FLOAT,
    damping_coefficient FLOAT,
    tonal_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Guitar body configurations
CREATE TABLE guitar_configurations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    name VARCHAR(100),
    body_shape ENUM('dreadnought', 'om', 'concert', 'grand_auditorium', 'jumbo', 'parlor'),
    scale_length FLOAT,
    bracing_pattern VARCHAR(50),
    top_tonewood_id INT REFERENCES tonewood_species(id),
    back_sides_tonewood_id INT REFERENCES tonewood_species(id),
    tonewood_embedding BLOB,
    frequency_response JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Microphone configurations
CREATE TABLE microphone_configurations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    guitar_config_id INT REFERENCES guitar_configurations(id),
    mic_model VARCHAR(50),
    position_distance FLOAT,
    position_angle FLOAT,
    position_target VARCHAR(50),
    stereo_technique ENUM('mono', 'xy', 'ab', 'ms', 'ortf'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Synthesis jobs
CREATE TABLE synthesis_jobs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    guitar_config_id INT REFERENCES guitar_configurations(id),
    mic_config_id INT REFERENCES microphone_configurations(id),
    input_type ENUM('midi', 'guitarroll', 'audio'),
    input_data LONGBLOB,
    output_url VARCHAR(500),
    status ENUM('pending', 'processing', 'completed', 'failed'),
    processing_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

### 8.2 TypeScript Type Definitions

```typescript
// Tonewood species definition
interface TonewoodSpecies {
  id: string;
  name: string;
  scientificName: string;
  citesStatus: 'appendix_i' | 'appendix_ii' | 'appendix_iii' | 'none';
  typicalUse: 'top' | 'back_sides' | 'neck' | 'bracing';
  physicalProperties: {
    densityMin: number;  // kg/m³
    densityMax: number;
    youngsModulusL: number;  // GPa
    youngsModulusR: number;
    shearModulus: number;
    soundVelocity: number;  // m/s
    dampingCoefficient: number;
  };
  tonalCharacteristics: {
    bassWeight: number;  // 0-1
    midrangePresence: number;
    trebleSparkle: number;
    sustain: number;
    overtoneComplexity: number;
  };
  description: string;
}

// Guitar configuration
interface GuitarConfiguration {
  id: string;
  name: string;
  bodyShape: 'dreadnought' | 'om' | 'concert' | 'grand_auditorium' | 'jumbo' | 'parlor';
  scaleLength: number;  // inches
  bracingPattern: string;
  topTonewood: TonewoodSpecies;
  backSidesTonewood: TonewoodSpecies;
  tonewoodEmbedding: number[];  // 256-dim vector
  predictedFrequencyResponse: FrequencyResponse;
}

// Microphone configuration
interface MicrophoneConfiguration {
  model: string;
  position: {
    distanceInches: number;
    angleDegrees: number;
    target: '12th_fret' | 'soundhole' | 'bridge' | 'body_front';
  };
  stereoTechnique: 'mono' | 'xy' | 'ab' | 'ms' | 'ortf';
}

// Frequency response
interface FrequencyResponse {
  frequenciesHz: number[];
  magnitudeDb: number[];
}

// Synthesis request
interface SynthesisRequest {
  guitarConfigId: string;
  micConfigId: string;
  input: {
    type: 'midi' | 'guitarroll' | 'audio';
    data: string;  // base64 encoded
  };
  outputFormat: {
    sampleRate: 44100 | 48000 | 96000;
    bitDepth: 16 | 24 | 32;
    channels: 'mono' | 'stereo';
  };
}
```

---

## 9. References

[1] Wegst, U.G.K. (2006). "Wood for Sound." American Journal of Botany, 93(10), 1439-1448. https://doi.org/10.3732/ajb.93.10.1439

[2] Badiane, D.G., Gonzalez, S., Malvermi, R., Antonacci, F., & Sarti, A. (2023). "A neural network-based method for spruce tonewood characterization." Journal of the Acoustical Society of America, 154(2), 730-738. https://pubs.aip.org/asa/jasa/article/154/2/730/2906397

[3] Viala, R., Placet, V., & Cogan, S. (2020). "Simultaneous non-destructive identification of multiple elastic and damping properties of spruce tonewood to improve grading." Journal of Cultural Heritage, 42, 108-116.

[4] Senior, M. (2010). "How To Record A Great Acoustic Guitar Sound." Sound on Sound. https://www.soundonsound.com/techniques/how-record-great-acoustic-guitar-sound

[5] Bedell Guitars. (2020). "Playing Brazilian rosewood is key to understanding its legendary allure." https://bedellguitars.com/playing-brazilian-rosewood-is-key-to-understanding-its-legendary-allure/

[6] Taylor Guitars. (2024). "Cocobolo." https://www.taylorguitars.com/guitars/acoustic/features/woods/body-woods/cocobolo

[7] Hayes, B., Shier, J., Fazekas, G., McPherson, A., & Sheridan, M. (2024). "A Review of Differentiable Digital Signal Processing for Music and Speech Synthesis." Frontiers in Signal Processing. https://www.frontiersin.org/articles/10.3389/frsip.2023.1284100

[8] Kim, H., Choi, S., & Nam, J. (2024). "Expressive Acoustic Guitar Sound Synthesis with an Instrument-Specific Input Representation and Diffusion Outpainting." ICASSP 2024. https://arxiv.org/abs/2401.13498

[9] Jonason, N. et al. (2024). "DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance from String-wise MIDI Input." DAFx 2024.

[10] Martin Guitar. (2024). "Wood & Materials." https://www.martinguitar.com/learn-wood-materials.html

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Complete
