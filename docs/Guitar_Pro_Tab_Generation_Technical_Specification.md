# Guitar Pro Tab Generation Technical Specification

## Starwood Audio-to-Tablature Pipeline

**Version**: 1.0  
**Author**: Manus AI  
**Date**: January 2026  
**Repository**: https://github.com/Mino01/Starwood

---

## Executive Summary

This technical specification outlines the architecture and implementation strategy for integrating **Guitar Pro tablature generation** into the Starwood music generation framework. The system will enable automatic transcription of AI-generated audio into playable guitar tablature, exported in industry-standard Guitar Pro formats (.gp, .gp5, .gpx).

The proposed pipeline consists of three primary stages: (1) audio-to-MIDI transcription using state-of-the-art neural networks, (2) MIDI-to-tablature conversion with playability optimization, and (3) Guitar Pro file generation using open-source libraries. This document provides a comprehensive analysis of available technologies, recommended architectures, and implementation guidelines.

---

## 1. Introduction

### 1.1 Background

Guitar tablature (commonly referred to as "tabs") is a form of musical notation that indicates instrument fingering rather than musical pitches. For guitarists, tablature provides an intuitive visual representation showing which fret to press on which string, making it significantly more accessible than traditional staff notation for learning specific songs [1].

The Guitar Pro format, developed by Arobas Music, has become the de facto standard for digital tablature distribution. Websites such as Ultimate Guitar host over 200,000 user-submitted Guitar Pro files, demonstrating the format's widespread adoption [2]. Integrating tablature generation into Starwood will enable users to not only generate music but also receive playable transcriptions suitable for practice and performance.

### 1.2 Objectives

The Guitar Pro Tab Generation module aims to achieve the following objectives:

1. **Automatic Transcription**: Convert Starwood-generated audio into accurate MIDI representations capturing pitch, timing, and dynamics.

2. **Intelligent Tablature Assignment**: Transform MIDI note data into guitar-specific string-fret combinations that prioritize playability and idiomatic fingering patterns.

3. **Multi-Format Export**: Generate output files compatible with Guitar Pro 3-8, MusicXML, and other standard notation formats.

4. **Real-Time Preview**: Provide web-based tablature visualization using the AlphaTab rendering engine.

---

## 2. System Architecture

### 2.1 Pipeline Overview

The tablature generation pipeline follows a three-stage architecture that separates concerns and allows for modular improvements:

| Stage | Component | Input | Output | Primary Technology |
|-------|-----------|-------|--------|-------------------|
| 1 | Audio Transcription | WAV/MP3 Audio | MIDI Events | Basic Pitch / Omnizart |
| 2 | Tablature Assignment | MIDI Events | Fretboard Positions | TabCNN / ML Model |
| 3 | File Generation | Fretboard Data | .gp5 / .gp File | PyGuitarPro |

### 2.2 Architectural Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STARWOOD AUDIO GENERATION                           │
│                    (RAVE-DDSP-Transformer Tri-Hybrid)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: AUDIO-TO-MIDI TRANSCRIPTION                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Basic Pitch    │    │    Omnizart     │    │   NeuralNote    │         │
│  │  (Primary)      │    │   (Fallback)    │    │   (Real-time)   │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           └──────────────────────┴──────────────────────┘                   │
│                                  │                                          │
│                                  ▼                                          │
│                         MIDI Event Stream                                   │
│              (pitch, onset, offset, velocity, pitch_bend)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: MIDI-TO-TABLATURE CONVERSION                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Probabilistic Fretboard DNN                       │   │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐        │   │
│  │  │ MIDI Encoder  │───▶│ Latent Space  │───▶│ Deconvolution │        │   │
│  │  │ (128 pitches) │    │   (384 dim)   │    │  (6×25 frets) │        │   │
│  │  └───────────────┘    └───────────────┘    └───────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Playability Optimizer                             │   │
│  │  • Maximum 6-fret stretch constraint                                 │   │
│  │  • Minimize finger travel distance                                   │   │
│  │  • Preserve fingering shapes between frames                          │   │
│  │  • Handle polyphonic input (max 6 simultaneous notes)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: GUITAR PRO FILE GENERATION                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  PyGuitarPro    │    │    AlphaTab     │    │   TuxGuitar     │         │
│  │  (.gp3-.gp5)    │    │  (Rendering)    │    │   (Editor)      │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│      .gp5 File            Web Preview             .gp7/.gp8 File           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Audio-to-MIDI Transcription

### 3.1 Technology Selection

The audio-to-MIDI transcription stage requires a neural network capable of polyphonic pitch detection with accurate onset and offset timing. After evaluating available options, we recommend **Spotify Basic Pitch** as the primary transcription engine due to its balance of accuracy, speed, and permissive licensing.

| Feature | Basic Pitch | Omnizart | NeuralNote |
|---------|-------------|----------|------------|
| **License** | Apache-2.0 | MIT | Proprietary |
| **Model Size** | <20 MB | ~500 MB | ~100 MB |
| **Real-time** | Yes | No | Yes |
| **Pitch Bend** | Yes | No | Limited |
| **Polyphonic** | Yes | Yes | Yes |
| **Guitar-specific** | No | No | No |

### 3.2 Basic Pitch Architecture

Basic Pitch employs a lightweight convolutional neural network that processes audio using a harmonic constant-Q transform (CQT) as input representation [3]. The model jointly predicts three outputs:

1. **Note Onsets**: Probability of a note beginning at each time-frequency bin
2. **Note Frames**: Probability of a note being active at each time-frequency bin  
3. **Contours**: Continuous pitch estimates for pitch bend detection

The architecture achieves state-of-the-art performance while maintaining a model size under 20 MB, enabling deployment in resource-constrained environments including web browsers via TensorFlow.js.

### 3.3 Implementation

```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

def transcribe_audio_to_midi(audio_path: str, output_path: str) -> dict:
    """
    Transcribe audio file to MIDI using Basic Pitch.
    
    Args:
        audio_path: Path to input audio file (WAV, MP3, FLAC)
        output_path: Path for output MIDI file
        
    Returns:
        Dictionary containing transcription metadata
    """
    model_output, midi_data, note_events = predict(
        audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58,  # milliseconds
        minimum_frequency=65.41,  # C2 (lowest guitar note in standard tuning)
        maximum_frequency=1318.51,  # E6 (highest practical guitar note)
    )
    
    midi_data.write(output_path)
    
    return {
        "note_count": len(note_events),
        "duration_seconds": midi_data.get_end_time(),
        "pitch_range": (min(n.pitch for n in note_events), 
                       max(n.pitch for n in note_events))
    }
```

### 3.4 Fallback Strategy

When Basic Pitch produces unsatisfactory results (e.g., for complex polyphonic passages), the system falls back to **Omnizart**, which provides specialized models for different instrument types [4]. Omnizart's music transcription module uses a more complex architecture that may capture nuances missed by the lighter Basic Pitch model.

---

## 4. Stage 2: MIDI-to-Tablature Conversion

### 4.1 The Tablature Assignment Problem

Converting MIDI pitches to guitar tablature is non-trivial because the same pitch can often be played at multiple positions on the fretboard. For example, the note E4 (MIDI pitch 64) can be played on:

- String 1 (high E), fret 0 (open)
- String 2 (B), fret 5
- String 3 (G), fret 9
- String 4 (D), fret 14
- String 5 (A), fret 19

The optimal choice depends on context: surrounding notes, hand position, musical style, and playability constraints [5]. This makes tablature assignment a sequence modeling problem well-suited to machine learning approaches.

### 4.2 Probabilistic Fretboard Neural Network

We adopt the architecture proposed in recent research on MIDI-to-tablature conversion [6], which uses a deep neural network to generate probabilistic fretboard representations. The network architecture consists of:

**Input Layer**: Concatenated binary vector of size 728:
- 128 dimensions for MIDI pitches to transcribe
- 600 dimensions for 4 previous tablature frames (4 × 6 strings × 25 frets)

**Encoder**: Three feedforward layers with scaled exponential linear (SELU) activation, producing a 384-dimensional latent representation.

**Decoder**: Two transposed convolution (deconvolution) layers that generate a 6×26 probabilistic fretboard (6 strings × 25 frets + open string).

### 4.3 Playability Constraints

The neural network output is refined using a search algorithm that enforces playability constraints:

1. **Single Note Per String**: At most one pitch can be assigned to each string simultaneously.

2. **Fret Stretch Limit**: All non-open-string notes must fall within a 6-fret window, reflecting realistic finger stretching capabilities.

3. **Motion Minimization**: The algorithm minimizes total finger travel distance between consecutive frames.

4. **Shape Preservation**: Common chord shapes and fingering patterns are preserved when transitioning between frames.

### 4.4 Implementation

```python
import torch
import torch.nn as nn

class ProbabilisticFretboardNet(nn.Module):
    """
    Neural network for MIDI-to-tablature conversion.
    Generates probabilistic fretboard representations from MIDI input.
    """
    
    def __init__(self, input_dim=728, latent_dim=384):
        super().__init__()
        
        # Encoder: MIDI + context → latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SELU(),
            nn.Linear(512, 448),
            nn.SELU(),
            nn.Linear(448, latent_dim),
            nn.SELU(),
        )
        
        # Decoder: latent space → probabilistic fretboard
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=(3, 6), stride=1),
            nn.SELU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 8), stride=(2, 2)),
            nn.Sigmoid(),
        )
        
    def forward(self, midi_input, previous_frames):
        # Concatenate MIDI input with previous tablature frames
        x = torch.cat([midi_input, previous_frames.flatten(1)], dim=1)
        
        # Encode to latent space
        latent = self.encoder(x)
        
        # Reshape for deconvolution
        latent = latent.view(-1, 384, 1, 1)
        
        # Decode to probabilistic fretboard (6 strings × 25 frets)
        fretboard = self.decoder(latent)
        
        return fretboard[:, :, :6, :25]  # Trim to valid fretboard dimensions


class PlayabilityOptimizer:
    """
    Converts probabilistic fretboard to playable tablature
    using constraint satisfaction and path optimization.
    """
    
    MAX_FRET_STRETCH = 6
    NUM_STRINGS = 6
    NUM_FRETS = 25
    
    def __init__(self, tuning="standard"):
        # Standard tuning: E2, A2, D3, G3, B3, E4
        self.string_pitches = [40, 45, 50, 55, 59, 64]
        
    def midi_to_fret_options(self, midi_pitch: int) -> list:
        """Get all possible string-fret combinations for a MIDI pitch."""
        options = []
        for string_idx, open_pitch in enumerate(self.string_pitches):
            fret = midi_pitch - open_pitch
            if 0 <= fret < self.NUM_FRETS:
                options.append((string_idx, fret))
        return options
    
    def optimize(self, prob_fretboard: torch.Tensor, 
                 previous_position: list) -> list:
        """
        Find optimal playable tablature from probabilistic fretboard.
        
        Returns:
            List of (string, fret) tuples for active notes
        """
        # Extract high-probability positions
        threshold = 0.5
        candidates = (prob_fretboard > threshold).nonzero()
        
        # Filter by playability constraints
        playable = self._filter_playable(candidates)
        
        # Optimize for minimal hand movement
        optimal = self._minimize_movement(playable, previous_position)
        
        return optimal
    
    def _filter_playable(self, candidates: list) -> list:
        """Filter candidates to ensure playability."""
        if len(candidates) > self.NUM_STRINGS:
            # Cannot play more than 6 notes simultaneously
            candidates = sorted(candidates, 
                              key=lambda x: x[1])[:self.NUM_STRINGS]
        
        # Check fret stretch constraint
        frets = [c[1] for c in candidates if c[1] > 0]
        if frets and (max(frets) - min(frets)) > self.MAX_FRET_STRETCH:
            # Remove outliers to satisfy stretch constraint
            candidates = self._remove_outliers(candidates)
        
        return candidates
    
    def _minimize_movement(self, candidates: list, 
                          previous: list) -> list:
        """Select positions that minimize hand movement."""
        if not previous:
            return candidates
            
        # Calculate center of previous position
        prev_frets = [p[1] for p in previous if p[1] > 0]
        prev_center = sum(prev_frets) / len(prev_frets) if prev_frets else 5
        
        # Score candidates by distance from previous center
        scored = []
        for c in candidates:
            distance = abs(c[1] - prev_center) if c[1] > 0 else 0
            scored.append((c, distance))
        
        # Return candidates sorted by minimal movement
        return [c for c, _ in sorted(scored, key=lambda x: x[1])]
```

### 4.5 Training Data

The model can be trained using the **DadaGP dataset**, which contains 26,181 Guitar Pro songs across 739 genres [2]. The dataset provides:

- Token-sequence representations suitable for sequence models
- Ground-truth tablature annotations
- Diverse musical styles for generalization

For audio-aligned training, the **GuitarSet dataset** provides 360 solo acoustic guitar recordings with time-aligned string and fret annotations [7].

---

## 5. Stage 3: Guitar Pro File Generation

### 5.1 File Format Support

Guitar Pro files have evolved through multiple versions, each with different internal structures:

| Version | Extension | Format Type | Library Support |
|---------|-----------|-------------|-----------------|
| GP3-GP5 | .gp3, .gp4, .gp5 | Binary | PyGuitarPro (full) |
| GP6 | .gpx | XML Archive | AlphaTab (read) |
| GP7-GP8 | .gp | ZIP/XML | AlphaTab (read), TuxGuitar |

### 5.2 PyGuitarPro Implementation

**PyGuitarPro** is a Python library that provides full read/write support for Guitar Pro 3-5 formats [8]. It exposes a comprehensive object model representing all elements of a Guitar Pro file:

```python
import guitarpro

def create_guitar_pro_file(tablature_data: list, 
                           metadata: dict,
                           output_path: str) -> None:
    """
    Generate a Guitar Pro file from tablature data.
    
    Args:
        tablature_data: List of measures, each containing beat/note data
        metadata: Song information (title, artist, tempo, etc.)
        output_path: Output file path (.gp5)
    """
    # Create new song
    song = guitarpro.models.Song()
    song.title = metadata.get("title", "Untitled")
    song.artist = metadata.get("artist", "Starwood AI")
    song.album = metadata.get("album", "")
    song.tempo = metadata.get("tempo", 120)
    
    # Create guitar track
    track = guitarpro.models.Track()
    track.name = "Guitar"
    track.channel.instrument = 25  # Acoustic Guitar (steel)
    track.isPercussionTrack = False
    
    # Set standard tuning (E A D G B E)
    track.strings = [
        guitarpro.models.GuitarString(1, 64),  # E4
        guitarpro.models.GuitarString(2, 59),  # B3
        guitarpro.models.GuitarString(3, 55),  # G3
        guitarpro.models.GuitarString(4, 50),  # D3
        guitarpro.models.GuitarString(5, 45),  # A2
        guitarpro.models.GuitarString(6, 40),  # E2
    ]
    
    # Add measures
    for measure_data in tablature_data:
        measure = guitarpro.models.Measure(track.measures[-1].header 
                                           if track.measures else None)
        
        for beat_data in measure_data["beats"]:
            beat = guitarpro.models.Beat(measure.voices[0])
            beat.duration.value = beat_data.get("duration", 4)  # Quarter note
            
            for note_data in beat_data.get("notes", []):
                note = guitarpro.models.Note(beat)
                note.string = note_data["string"]
                note.value = note_data["fret"]
                note.velocity = note_data.get("velocity", 95)
                
                # Add effects if present
                if "bend" in note_data:
                    note.effect.bend = create_bend_effect(note_data["bend"])
                if "slide" in note_data:
                    note.effect.slides.append(note_data["slide"])
                if "hammer" in note_data:
                    note.effect.hammer = note_data["hammer"]
                
                beat.notes.append(note)
            
            measure.voices[0].beats.append(beat)
        
        track.measures.append(measure)
    
    song.tracks.append(track)
    
    # Write to file
    guitarpro.write(song, output_path)


def create_bend_effect(bend_data: dict) -> guitarpro.models.BendEffect:
    """Create a pitch bend effect from bend data."""
    bend = guitarpro.models.BendEffect()
    bend.type = guitarpro.models.BendType.bend
    bend.value = bend_data.get("semitones", 1) * 50  # GP uses 50 units per semitone
    
    # Add bend points for smooth curve
    bend.points = [
        guitarpro.models.BendPoint(0, 0),
        guitarpro.models.BendPoint(6, bend.value),
        guitarpro.models.BendPoint(12, bend.value),
    ]
    
    return bend
```

### 5.3 Web-Based Rendering with AlphaTab

For web-based tablature preview, **AlphaTab** provides a comprehensive JavaScript library that renders Guitar Pro files directly in the browser [9]. Integration with the Starwood web application:

```typescript
import { AlphaTabApi, Settings } from '@coderline/alphatab';

export class TablatureViewer {
    private api: AlphaTabApi;
    
    constructor(container: HTMLElement) {
        const settings: Settings = {
            core: {
                engine: 'html5',
                logLevel: 1,
            },
            display: {
                staveProfile: 'Tab',
                resources: {
                    staffLineColor: '#999999',
                    barSeparatorColor: '#999999',
                },
            },
            notation: {
                elements: {
                    scoreTitle: true,
                    scoreArtist: true,
                    effectTempo: true,
                },
            },
            player: {
                enablePlayer: true,
                enableCursor: true,
                enableUserInteraction: true,
                soundFont: '/soundfonts/default.sf2',
            },
        };
        
        this.api = new AlphaTabApi(container, settings);
    }
    
    async loadFile(fileUrl: string): Promise<void> {
        await this.api.load(fileUrl);
    }
    
    async loadFromArrayBuffer(buffer: ArrayBuffer): Promise<void> {
        await this.api.load(new Uint8Array(buffer));
    }
    
    play(): void {
        this.api.play();
    }
    
    pause(): void {
        this.api.pause();
    }
    
    setTempo(bpm: number): void {
        this.api.playbackSpeed = bpm / this.api.score.tempo;
    }
}
```

---

## 6. Integration with Starwood

### 6.1 API Endpoints

The tablature generation module exposes the following API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tablature/transcribe` | POST | Transcribe audio to MIDI |
| `/api/v1/tablature/convert` | POST | Convert MIDI to tablature |
| `/api/v1/tablature/generate` | POST | Full pipeline (audio → GP file) |
| `/api/v1/tablature/preview` | GET | Get tablature preview data |
| `/api/v1/tablature/export` | GET | Download GP file |

### 6.2 Data Flow

```
User Request (audio_url, options)
         │
         ▼
┌─────────────────────────────────────┐
│     Starwood Tablature Service     │
│  ┌─────────────────────────────┐    │
│  │  1. Download/validate audio │    │
│  │  2. Transcribe to MIDI      │    │
│  │  3. Convert to tablature    │    │
│  │  4. Generate GP file        │    │
│  │  5. Store in S3             │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         │
         ▼
Response (gp_file_url, preview_data, metadata)
```

### 6.3 Database Schema

```sql
CREATE TABLE tablature_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    audio_url VARCHAR(512) NOT NULL,
    status ENUM('pending', 'transcribing', 'converting', 'generating', 'completed', 'failed') DEFAULT 'pending',
    midi_url VARCHAR(512),
    gp_file_url VARCHAR(512),
    metadata JSON,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE tablature_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_id INT NOT NULL,
    tuning VARCHAR(32) DEFAULT 'standard',
    capo_fret INT DEFAULT 0,
    tempo_override INT,
    time_signature VARCHAR(8) DEFAULT '4/4',
    include_drums BOOLEAN DEFAULT FALSE,
    include_bass BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (job_id) REFERENCES tablature_jobs(id)
);
```

---

## 7. Performance Considerations

### 7.1 Processing Time Estimates

| Stage | Typical Duration | Factors |
|-------|------------------|---------|
| Audio Transcription | 0.5-2× real-time | Audio length, polyphony |
| Tablature Conversion | 0.1-0.5× real-time | Note density, complexity |
| GP File Generation | <1 second | File size |
| **Total Pipeline** | **1-3× real-time** | Combined factors |

### 7.2 Optimization Strategies

1. **GPU Acceleration**: Deploy Basic Pitch and the fretboard neural network on GPU for 5-10× speedup.

2. **Batch Processing**: Process multiple audio segments in parallel for long recordings.

3. **Caching**: Cache MIDI transcriptions to avoid re-processing when only tablature parameters change.

4. **Progressive Loading**: Stream tablature data to the frontend as it becomes available.

---

## 8. Future Enhancements

### 8.1 Planned Features

1. **Multi-Instrument Support**: Extend tablature generation to bass guitar, ukulele, and other fretted instruments.

2. **Style-Aware Fingering**: Train specialized models for different genres (metal power chords, jazz voicings, classical fingerstyle).

3. **Aurora LLM Integration**: Use Aurora LLM to interpret natural language descriptions of desired playing style and incorporate into tablature generation.

4. **Real-Time Transcription**: Enable live audio input with streaming tablature output for practice applications.

### 8.2 Research Directions

1. **End-to-End Audio-to-Tab**: Train a single neural network that directly maps audio spectrograms to tablature, bypassing the MIDI intermediate representation.

2. **Expressive Technique Detection**: Improve detection of guitar-specific techniques (hammer-ons, pull-offs, slides, bends, vibrato) directly from audio.

3. **Arrangement Optimization**: Automatically simplify complex passages for different skill levels while preserving musical intent.

---

## 9. References

[1] P. Sarmento et al., "DadaGP: A Dataset of Tokenized GuitarPro Songs for Sequence Models," *ISMIR 2021*. https://archives.ismir.net/ismir2021/paper/000076.pdf

[2] DadaGP GitHub Repository. https://github.com/dada-bots/dadaGP

[3] Spotify Basic Pitch. https://github.com/spotify/basic-pitch

[4] Y.-T. Wu et al., "Omnizart: A General Toolbox for Automatic Music Transcription," *Journal of Open Source Software*, vol. 6, no. 68, 2021. https://github.com/Music-and-Culture-Technology-Lab/omnizart

[5] "A Machine Learning Approach for MIDI to Guitar Tablature Conversion," *arXiv:2510.10619*, 2025. https://arxiv.org/abs/2510.10619

[6] A. Wiggins and Y.-E. Kim, "Guitar Tablature Estimation with a Convolutional Neural Network," *ISMIR 2019*. https://archives.ismir.net/ismir2019/paper/000033.pdf

[7] Q. Xi et al., "GuitarSet: A Dataset for Guitar Transcription," *ISMIR 2018*. https://guitarset.weebly.com/

[8] PyGuitarPro Documentation. https://pyguitarpro.readthedocs.io/

[9] AlphaTab Documentation. https://alphatab.net/docs/introduction

---

## Appendix A: Supported Guitar Tunings

| Tuning Name | Strings (Low to High) | MIDI Pitches |
|-------------|----------------------|--------------|
| Standard | E A D G B E | 40 45 50 55 59 64 |
| Drop D | D A D G B E | 38 45 50 55 59 64 |
| Half-Step Down | Eb Ab Db Gb Bb Eb | 39 44 49 54 58 63 |
| Full-Step Down | D G C F A D | 38 43 48 53 57 62 |
| Open G | D G D G B D | 38 43 50 55 59 62 |
| Open D | D A D F# A D | 38 45 50 54 57 62 |
| DADGAD | D A D G A D | 38 45 50 55 57 62 |

## Appendix B: Guitar Pro Effect Codes

| Effect | GP Code | Description |
|--------|---------|-------------|
| Hammer-on | `hammer = True` | Legato note without picking |
| Pull-off | `pullOff = True` | Legato note pulling finger off |
| Slide In | `slides = [SlideType.intoFromAbove]` | Slide into note from above |
| Slide Out | `slides = [SlideType.outDownwards]` | Slide out of note downwards |
| Bend | `bend.type = BendType.bend` | Pitch bend effect |
| Vibrato | `vibrato = True` | Vibrato effect |
| Palm Mute | `palmMute = True` | Palm-muted note |
| Let Ring | `letRing = True` | Note sustains through subsequent beats |
| Harmonic | `harmonic.type = HarmonicType.natural` | Natural harmonic |
| Tap | `tapping = True` | Two-hand tapping technique |

