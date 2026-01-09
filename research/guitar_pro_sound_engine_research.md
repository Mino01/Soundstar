# Starwood Sound Engine for Guitar Pro Files - Research Notes

## 1. Guitar Pro File Format Structure

### File Versions
- **GP3**: Guitar Pro 3 format (basic)
- **GP4**: Guitar Pro 4 format (enhanced chords)
- **GP5**: Guitar Pro 5 format (RSE support)
- **GP6**: XML-based format
- **GP7/GP8**: ZIP-based archive format

### Core Data Elements (from PyGuitarPro documentation)

#### Song Structure
- Score information (title, artist, album, etc.)
- Triplet feel
- Tempo (BPM)
- Key signature
- MIDI channels (64 channels: 4 ports × 16 channels)
- Measure headers
- Tracks
- Measures

#### Track Information
- Track flags (drums, 12-string, banjo)
- Track name (40 characters)
- Number of strings
- String tuning (7 integers, highest to lowest)
- MIDI port
- MIDI channel
- Number of frets
- Capo position
- Track color

#### MIDI Channel Data
- Instrument (MIDI program)
- Volume
- Balance (pan)
- Chorus
- Reverb
- Phaser
- Tremolo

#### Beat/Note Data
- Beat flags (dotted, chord, text, effects, mix table, tuplet, rest)
- Duration (-2=whole, -1=half, 0=quarter, 1=eighth, 2=sixteenth, 3=thirty-second)
- Chord diagrams
- Beat effects (vibrato, harmonic, fade in, tremolo bar, stroke direction)
- Mix table changes

#### Note Effects
- Vibrato (normal and wide)
- Natural harmonic
- Artificial harmonic
- Fade in
- Tremolo bar (dip)
- Tapping, slap, pop
- Beat stroke direction
- Bend effects
- Slide effects
- Hammer-on/pull-off
- Let ring
- Palm mute
- Staccato
- Ghost notes
- Accent

### String/Fret Encoding
- Strings encoded as bit flags (0x01=7th string, 0x02=6th, etc.)
- Fret values stored per string
- -1 = string not played

## 2. RSE (Realistic Sound Engine) Technology

### What is RSE?
- Guitar Pro's proprietary sound engine
- Uses recorded samples of real instruments
- Includes amp/effect modeling
- Soundbanks installed separately

### RSE Components
1. **Instrument Samples**: Multi-sampled recordings of guitars, bass, drums
2. **Amp Modeling**: Simulated amplifier tones
3. **Effect Pedals**: Distortion, overdrive, delay, reverb, etc.
4. **Effect Racks**: Studio effects (EQ, compression, etc.)
5. **Signature Sounds**: Pre-configured artist presets

### RSE Limitations
- Large file sizes (soundbanks)
- Limited customization
- Sample-based (not physically modeled)
- Cannot capture nuanced playing dynamics

## 3. Key Data for Sound Engine Implementation

### Essential Parameters for Rendering
1. **Note Information**
   - Fret position
   - String number
   - Duration
   - Velocity/dynamics

2. **Articulation Data**
   - Hammer-on/pull-off
   - Slide (legato slide, shift slide)
   - Bend (full, half, quarter, pre-bend, release)
   - Vibrato
   - Tapping
   - Harmonics (natural, artificial, pinch, semi)
   - Palm mute
   - Let ring
   - Staccato

3. **Timing Information**
   - Tempo
   - Time signature
   - Tuplets
   - Grace notes

4. **Instrument Setup**
   - Tuning
   - Capo position
   - Number of strings
   - Instrument type (acoustic, electric, bass, etc.)

5. **Effects Chain**
   - Amp settings
   - Pedal effects
   - Rack effects
   - Mix settings (volume, pan, reverb, etc.)

## 4. Libraries for Reading Guitar Pro Files

### PyGuitarPro (Python)
- License: LGPL-3.0
- Supports: GP3, GP4, GP5
- Full read/write support
- Well-documented API

### AlphaTab (JavaScript/TypeScript)
- License: MPL-2.0
- Supports: GP3, GP4, GP5, GP6, GP7, GP8
- Web rendering engine
- Cross-platform

## Next Steps
- Research sample-based vs synthesis-based guitar rendering
- Investigate DDSP for guitar sound synthesis
- Study existing open-source guitar synthesizers
- Design Starwood sound engine architecture


## 5. DDSP-Based Neural Waveform Synthesis for Guitar

### Key Research: Jonason et al. (2023)
**Paper**: "DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance from String-wise MIDI Input"
**Authors**: Nicolas Jonason, Xin Wang, Erica Cooper, Lauri Juvela, Bob L. T. Sturm, Junichi Yamagishi
**Institutions**: KTH Royal Institute of Technology, National Institute of Informatics Japan, Aalto University

### Key Findings

#### Challenge: Polyphonic Guitar Synthesis
Unlike piano where each key produces independent sound, guitar strings interact. Multiple voices can be active simultaneously with varying pitch due to bends, slides, and other articulations. Guitar performance involves idiosyncrasies like bending strings, sliding, and articulations where pitch is not discrete.

#### Proposed Systems (4 configurations)
1. **Default Configuration (ctr→syn→rg)**: Control-synthesis architecture
   - Takes string-wise MIDI input
   - Predicts four control features per string:
     - Fundamental frequency (F0)
     - Loudness
     - Periodicity
     - Spectral centroid
   - Uses synthesis decoder + harmonic-noise-reverb synthesizer

2. **Classification vs Regression**: Formulating control feature prediction as classification (quantized bins) yields better results than regression

3. **Simplest System (Best Performance)**: Directly predicts synthesis parameters from MIDI input without intermediate control features

#### Technical Details
- **Input**: String-wise MIDI (6 strings for standard guitar)
- **Pitch bins**: 305 (quantized pitch values)
- **Loudness bins**: 64
- **Periodicity bins**: 6
- **Feature frame rate**: Configurable for different render durations

#### Architecture Components
- **Synthesis Decoder**: Neural network predicting synthesis parameters
- **Harmonic-Noise-Reverb Synthesizer**: DDSP-based audio generation
- **Control-Synthesis Approach**: Adapted from speech synthesis

### Implications for Starwood Sound Engine
This research directly applies to implementing a Guitar Pro sound engine:
1. String-wise MIDI input maps directly to Guitar Pro tablature data
2. DDSP synthesis can generate realistic acoustic guitar sounds
3. Control features (F0, loudness, periodicity) can be derived from tab notation
4. Classification-based prediction is more effective than regression


## 6. Physical Modeling Guitar Synthesis

### Research: CUHK Guitar Virtual Instrument (ICMC 2020)
**Project**: Guitar Virtual Instrument using Physical Modelling with Collision Simulation
**Institution**: Chinese University of Hong Kong
**Authors**: Ho Ka Wing, Ling Yiu, Chau Chuk Jee

### Technical Approach
Physical modeling synthesis simulates the actual physics of guitar strings and body:

1. **Finite Difference Method**: Each string segment is individually simulated
2. **Collision Simulation**: Models fret collision, finger collision
3. **Damping**: Realistic energy loss over time
4. **Stiffness**: String stiffness affects harmonics
5. **Variable Tension**: Pitch bends, vibrato simulation

### Supported Articulations
- Sustain
- Palm mute
- Tapping
- Slapping
- Sliding (legato slide)
- Natural harmonics
- Fingerpicking vs plectrum

### Customization Parameters
- Up to 12 strings supported
- 240+ controllable parameters
- Acoustic and electric guitar modes
- Fretted and fretless options

### Performance Optimization
- SIMD acceleration
- Multi-threading
- Minimal RAM/storage usage
- Real-time capable

### Implications for Starwood
Physical modeling offers:
- True physics-based sound generation
- Highly customizable guitar characteristics
- Realistic articulation handling
- No sample library required
- Computationally intensive but achievable with modern hardware

## 7. Comparison: Sample-Based vs Synthesis-Based Approaches

| Aspect | Sample-Based (RSE) | DDSP Neural | Physical Modeling |
|--------|-------------------|-------------|-------------------|
| **Realism** | High (recorded) | High (learned) | High (simulated) |
| **File Size** | Large (GB) | Medium (MB) | Small (KB) |
| **Flexibility** | Limited | Moderate | Very High |
| **Articulations** | Pre-recorded | Learned | Simulated |
| **Customization** | Low | Moderate | Very High |
| **CPU Usage** | Low | Medium | High |
| **Latency** | Low | Medium | Variable |
| **Tonewood Modeling** | Fixed | Possible | Native |

### Recommendation for Starwood
A **hybrid approach** combining:
1. **DDSP for base tone generation** - Efficient, high-quality
2. **Physical modeling for articulations** - Realistic bends, slides, harmonics
3. **Neural tonewood transformation** - Apply Starwood's unique tonewood character


## 8. Commercial Guitar VST Articulation Handling

### Case Study: Shreddage 3 (Impact Soundworks)

#### Articulations Supported
- Sustains
- Powerchords
- Palm mutes (3 layers)
- Staccatos
- Harmonics (natural, artificial)
- Tapping
- Tremolo
- Slides
- Fingered vibrato
- FX
- Hammer-on / Pull-off
- Rakes
- Chokes

#### Sampling Approach
- Over 25,000 pristine 24-bit samples
- Every string sampled from open to highest fret
- Up to 4x down/4x up strokes per note
- 3 dynamic layers (velocity-based)
- 3 palm mute layers
- Three pickup positions (neck, middle, bridge) recorded DI

#### TACT (Total Articulation Control Technology)
Allows customization of how articulations are mapped and triggered:
- Keyswitch-based articulation switching
- Velocity-based articulation selection
- MIDI CC-based control
- Customizable per-DAW setups

#### Console FX System
- 30 effects modules
- EQs (digital and analog-style)
- Compressors
- Spatial FX
- Modulation FX
- Amp simulations
- Distortion pedals
- Reverbs
- 30+ custom cabinet IRs

### Implications for Starwood Guitar Pro Sound Engine
The Shreddage approach demonstrates:
1. **Multi-layer sampling** is key for realism (velocity, stroke direction, palm mute depth)
2. **Articulation mapping** must be flexible and customizable
3. **FX chain integration** is essential for production-ready sound
4. **Pickup position blending** adds tonal versatility

## 9. Tablature-to-Audio Rendering Pipeline

### Data Flow: Guitar Pro → Audio

```
Guitar Pro File (.gp5/.gp7)
        ↓
    PyGuitarPro Parser
        ↓
    Tablature Data Model
    ├── Track Info (tuning, capo, strings)
    ├── Measure/Beat Structure
    ├── Note Data (fret, string, duration)
    └── Articulation Effects
        ↓
    MIDI-like Event Stream
    ├── Note On/Off events
    ├── Pitch (fret + string → MIDI note)
    ├── Velocity (dynamics)
    └── Articulation markers
        ↓
    Starwood Sound Engine
    ├── Tonewood Character Module
    ├── DDSP Synthesis Core
    ├── Articulation Processor
    └── FX Chain
        ↓
    Audio Output (WAV/MP3)
```

### Key Conversion Challenges

1. **Fret/String to Pitch**: Must account for tuning and capo
2. **Duration to Samples**: Convert beat duration to sample count at given tempo
3. **Articulation Mapping**: Map GP effects to synthesis parameters
4. **Dynamics**: Infer velocity from notation (accent, ghost note, etc.)
5. **Timing**: Handle tuplets, grace notes, tempo changes

### Articulation Effect Mapping

| Guitar Pro Effect | Synthesis Parameter |
|-------------------|---------------------|
| Palm Mute | Damping coefficient, filter cutoff |
| Hammer-on/Pull-off | Legato mode, attack reduction |
| Slide | Pitch glide, portamento |
| Bend | Pitch envelope |
| Vibrato | Pitch LFO |
| Harmonic | Harmonic mode, filter |
| Let Ring | Extended release |
| Staccato | Short release |
| Tremolo Picking | Rapid retriggering |
| Tapping | Velocity reduction, legato |

