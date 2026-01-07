# GPT Codex Integration with MusicAI Tri-Hybrid Engine

## Technical Specification Document

**Author**: Manus AI  
**Version**: 1.0  
**Date**: January 7, 2026  
**Status**: Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [API Endpoints Design](#3-api-endpoints-design)
4. [Natural Language to Musical Parameters Workflow](#4-natural-language-to-musical-parameters-workflow)
5. [Integration Points: Codex, Aurora LLM, and Control Encoder](#5-integration-points-codex-aurora-llm-and-control-encoder)
6. [Data Models and Schemas](#6-data-models-and-schemas)
7. [Real-Time Prompt Refinement Implementation](#7-real-time-prompt-refinement-implementation)
8. [Error Handling and Fallback Strategies](#8-error-handling-and-fallback-strategies)
9. [Performance Optimization Strategies](#9-performance-optimization-strategies)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

This technical specification document outlines the integration of **GPT Codex** (OpenAI's code-generation and natural language understanding model) with the **MusicAI Tri-Hybrid Engine**. The primary objective is to enable **natural language-driven music generation** with **intelligent prompt enhancement**, allowing users to describe their musical vision in plain English and receive high-fidelity, semantically coherent audio output.

The integration leverages Codex's advanced natural language understanding capabilities to:

1. **Parse and interpret** complex, nuanced musical descriptions from users.
2. **Enhance user prompts** by adding musical context, suggesting instrumentation, and refining structural elements.
3. **Translate natural language** into structured musical parameters compatible with the MusicAI Control Encoder.
4. **Collaborate with Aurora LLM** to provide a multi-layered semantic understanding pipeline.

This document provides a complete technical blueprint for developers to implement the Codex integration seamlessly into the existing MusicAI system.

---

## 2. Architecture Overview

### 2.1 High-Level System Architecture

The GPT Codex integration introduces a new **Semantic Intelligence Layer** that sits between the user interface and the MusicAI Tri-Hybrid Engine's Control Encoder. This layer acts as an intelligent intermediary, processing raw user input and producing enriched, structured musical parameters.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  (Web UI / API Client)                                                       │
│  - Natural Language Input                                                    │
│  - Real-time Prompt Suggestions                                              │
│  - Parameter Refinement Controls                                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEMANTIC INTELLIGENCE LAYER                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         GPT CODEX MODULE                             │    │
│  │  - Prompt Enhancement Engine                                         │    │
│  │  - Musical Context Analyzer                                          │    │
│  │  - Structured Parameter Generator                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         AURORA LLM MODULE                            │    │
│  │  - Semantic Deconstruction                                           │    │
│  │  - Structural Guidance                                               │    │
│  │  - Musical Knowledge Base                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    UNIFIED CONTROL EMBEDDING                         │    │
│  │  - Genre, Tempo, Mood, Instrumentation                               │    │
│  │  - Structural Form (AABA, Verse/Chorus, etc.)                        │    │
│  │  - f0, Loudness, Harmonic Mix Parameters                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MUSICAI TRI-HYBRID ENGINE                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │  Structural   │───▶│  RAVE Codec   │───▶│    DDSP       │                │
│  │  Transformer  │    │  (Latent)     │    │  Refinement   │                │
│  └───────────────┘    └───────────────┘    └───────────────┘                │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌───────────────┐                                   │
│                          │   Ultimate    │                                   │
│                          │    Mixer      │                                   │
│                          └───────────────┘                                   │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌───────────────┐                                   │
│                          │  Final Audio  │                                   │
│                          │   Output      │                                   │
│                          └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| **GPT Codex Module** | Parses natural language, enhances prompts, generates structured suggestions | Raw user text prompt | Enhanced prompt, structured musical parameters (JSON) |
| **Aurora LLM Module** | Provides deep semantic deconstruction, structural guidance, and musical knowledge | Enhanced prompt from Codex | Unified Control Embedding vector |
| **Control Encoder** | Combines all semantic information into a single embedding | Codex parameters + Aurora guidance | `control_embedding` tensor |
| **Structural Transformer** | Generates RAVE latent vector from control embedding | `control_embedding` | `z_rave` (latent vector) |
| **RAVE Codec** | Decodes latent vector to high-fidelity audio waveform | `z_rave` | `w_rave` (audio waveform) |
| **DDSP Refinement Net** | Provides interpretable timbre control | `z_rave` | `w_ddsp` (refined audio), `p_ddsp` (parameters) |
| **Ultimate Mixer** | Blends RAVE and DDSP outputs | `w_rave`, `w_ddsp`, `p_ddsp` | `w_final` (final audio) |

### 2.3 Data Flow Sequence

The following sequence diagram illustrates the data flow for a typical music generation request:

```
User                 WebUI/API          Codex Module        Aurora LLM         Control Encoder      Tri-Hybrid Engine
  │                      │                   │                   │                   │                      │
  │──(1) NL Prompt──────▶│                   │                   │                   │                      │
  │                      │──(2) Enhance──────▶│                   │                   │                      │
  │                      │                   │──(3) API Call─────▶│ (OpenAI Codex)   │                      │
  │                      │                   │◀──(4) Enhanced────│                   │                      │
  │◀──(5) Suggestions───│◀──────────────────│                   │                   │                      │
  │──(6) Confirm/Refine─▶│                   │                   │                   │                      │
  │                      │──(7) Deconstruct──────────────────────▶│                   │                      │
  │                      │                   │                   │──(8) Semantic────▶│                      │
  │                      │                   │                   │   Embedding       │                      │
  │                      │                   │                   │                   │──(9) Generate───────▶│
  │                      │                   │                   │                   │                      │
  │                      │                   │                   │                   │◀──(10) Audio────────│
  │◀──(11) Final Audio──│◀──────────────────────────────────────────────────────────│                      │
  │                      │                   │                   │                   │                      │
```

---

## 3. API Endpoints Design

This section defines the RESTful API endpoints for the Codex-powered music generation and prompt enhancement features. All endpoints are prefixed with `/api/v1/codex`.

### 3.1 Endpoint Summary

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/api/v1/codex/enhance` | Enhance a raw user prompt with musical context | Yes |
| `POST` | `/api/v1/codex/suggest` | Get real-time suggestions as user types | Yes |
| `POST` | `/api/v1/codex/translate` | Translate enhanced prompt to structured parameters | Yes |
| `POST` | `/api/v1/codex/generate` | Full pipeline: enhance, translate, and generate audio | Yes |
| `GET`  | `/api/v1/codex/history` | Retrieve user's prompt enhancement history | Yes |
| `POST` | `/api/v1/codex/feedback` | Submit feedback on Codex suggestions for model improvement | Yes |

### 3.2 Endpoint Specifications

#### 3.2.1 `POST /api/v1/codex/enhance`

Enhances a raw user prompt by adding musical context, suggesting instrumentation, and refining structural elements.

**Request Body:**

```json
{
  "prompt": "string (required) - The raw natural language prompt from the user",
  "context": {
    "genre_hint": "string (optional) - A hint about the desired genre",
    "mood_hint": "string (optional) - A hint about the desired mood",
    "duration_hint": "number (optional) - Desired duration in seconds"
  },
  "enhancement_level": "string (optional) - 'minimal', 'moderate', 'aggressive'. Default: 'moderate'"
}
```

**Response Body (Success - 200 OK):**

```json
{
  "original_prompt": "string - The original user prompt",
  "enhanced_prompt": "string - The Codex-enhanced prompt",
  "suggestions": [
    {
      "type": "string - 'genre', 'instrumentation', 'mood', 'structure', 'tempo'",
      "original_value": "string | null - The inferred original value",
      "suggested_value": "string - The suggested value",
      "confidence": "number - Confidence score (0.0 - 1.0)",
      "reasoning": "string - Explanation for the suggestion"
    }
  ],
  "structured_parameters": {
    "genre": "string",
    "sub_genre": "string | null",
    "tempo_bpm": "number",
    "mood": "string",
    "energy_level": "number (0.0 - 1.0)",
    "instrumentation": ["string"],
    "structure": "string - e.g., 'AABA', 'Verse-Chorus-Verse'",
    "key_signature": "string | null",
    "time_signature": "string | null"
  },
  "codex_metadata": {
    "model_version": "string",
    "processing_time_ms": "number",
    "tokens_used": "number"
  }
}
```

**Error Responses:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_PROMPT` | Prompt is empty or exceeds maximum length |
| 401 | `UNAUTHORIZED` | Authentication token is missing or invalid |
| 429 | `RATE_LIMITED` | Too many requests, try again later |
| 503 | `CODEX_UNAVAILABLE` | Codex API is temporarily unavailable |

#### 3.2.2 `POST /api/v1/codex/suggest`

Provides real-time suggestions as the user types, enabling an interactive prompt refinement experience.

**Request Body:**

```json
{
  "partial_prompt": "string (required) - The current partial prompt being typed",
  "cursor_position": "number (optional) - Cursor position in the prompt",
  "suggestion_count": "number (optional) - Number of suggestions to return. Default: 3, Max: 5"
}
```

**Response Body (Success - 200 OK):**

```json
{
  "suggestions": [
    {
      "text": "string - The suggested completion or addition",
      "type": "string - 'completion', 'addition', 'replacement'",
      "insert_position": "number - Position to insert the suggestion",
      "confidence": "number - Confidence score (0.0 - 1.0)"
    }
  ],
  "inferred_context": {
    "likely_genre": "string | null",
    "likely_mood": "string | null"
  }
}
```

#### 3.2.3 `POST /api/v1/codex/translate`

Translates an enhanced prompt (or raw prompt) into structured musical parameters compatible with the MusicAI Control Encoder.

**Request Body:**

```json
{
  "prompt": "string (required) - The prompt to translate (can be raw or enhanced)",
  "target_format": "string (optional) - 'control_encoder', 'midi_params', 'full'. Default: 'control_encoder'"
}
```

**Response Body (Success - 200 OK):**

```json
{
  "control_encoder_params": {
    "genre_embedding": [0.1, 0.2, ...],
    "tempo_normalized": 0.65,
    "mood_embedding": [0.3, 0.4, ...],
    "instrumentation_embedding": [0.5, 0.6, ...],
    "structure_embedding": [0.7, 0.8, ...],
    "energy_level": 0.75,
    "valence": 0.6,
    "danceability": 0.8
  },
  "human_readable_params": {
    "genre": "Electronic / Synthwave",
    "tempo_bpm": 120,
    "mood": "Energetic, Nostalgic",
    "instrumentation": ["Synthesizer", "Drum Machine", "Bass"],
    "structure": "Intro-Verse-Chorus-Verse-Chorus-Outro",
    "key": "A minor",
    "time_signature": "4/4"
  },
  "ddsp_hints": {
    "f0_range": [80, 800],
    "loudness_target": 0.7,
    "harmonic_mix": 0.6
  }
}
```

#### 3.2.4 `POST /api/v1/codex/generate`

A convenience endpoint that combines prompt enhancement, translation, and audio generation into a single request.

**Request Body:**

```json
{
  "prompt": "string (required) - The raw natural language prompt",
  "options": {
    "enhance_prompt": "boolean (optional) - Whether to enhance the prompt. Default: true",
    "duration_seconds": "number (optional) - Desired audio duration. Default: 30",
    "output_format": "string (optional) - 'wav', 'mp3', 'flac'. Default: 'wav'",
    "sample_rate": "number (optional) - Output sample rate. Default: 44100"
  }
}
```

**Response Body (Success - 200 OK):**

```json
{
  "generation_id": "string - Unique identifier for this generation",
  "status": "string - 'completed', 'processing', 'queued'",
  "audio_url": "string - URL to download the generated audio",
  "enhanced_prompt": "string - The enhanced prompt used for generation",
  "parameters_used": { ... },
  "generation_metadata": {
    "total_time_ms": "number",
    "codex_time_ms": "number",
    "aurora_time_ms": "number",
    "engine_time_ms": "number"
  }
}
```

#### 3.2.5 `GET /api/v1/codex/history`

Retrieves the user's prompt enhancement and generation history.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | number | Number of records to return. Default: 20, Max: 100 |
| `offset` | number | Offset for pagination. Default: 0 |
| `sort` | string | Sort order: 'newest', 'oldest'. Default: 'newest' |

**Response Body (Success - 200 OK):**

```json
{
  "total_count": "number",
  "items": [
    {
      "id": "string",
      "original_prompt": "string",
      "enhanced_prompt": "string",
      "parameters": { ... },
      "audio_url": "string | null",
      "created_at": "string (ISO 8601)",
      "user_feedback": "string | null - 'positive', 'negative', 'neutral'"
    }
  ]
}
```

#### 3.2.6 `POST /api/v1/codex/feedback`

Submits user feedback on Codex suggestions to improve future recommendations.

**Request Body:**

```json
{
  "generation_id": "string (required) - The ID of the generation to provide feedback on",
  "feedback_type": "string (required) - 'positive', 'negative', 'neutral'",
  "feedback_details": {
    "prompt_quality": "number (optional) - 1-5 rating",
    "suggestion_relevance": "number (optional) - 1-5 rating",
    "audio_quality": "number (optional) - 1-5 rating",
    "comments": "string (optional) - Free-form feedback"
  }
}
```

**Response Body (Success - 200 OK):**

```json
{
  "success": true,
  "message": "Feedback submitted successfully"
}
```

---

## 4. Natural Language to Musical Parameters Workflow

This section documents the detailed workflow for translating natural language descriptions into structured musical parameters using GPT Codex.

### 4.1 Workflow Overview

The translation process consists of five distinct stages:

1. **Input Preprocessing**: Normalize and clean the raw user input.
2. **Semantic Parsing**: Use Codex to extract musical intent and entities.
3. **Parameter Mapping**: Map extracted entities to MusicAI's parameter schema.
4. **Validation and Defaulting**: Validate parameters and apply sensible defaults.
5. **Embedding Generation**: Convert parameters to numerical embeddings for the Control Encoder.

### 4.2 Stage 1: Input Preprocessing

Before sending the prompt to Codex, the system performs the following preprocessing steps:

| Step | Description | Example |
|------|-------------|---------|
| **Normalization** | Convert to lowercase, remove excessive whitespace | "  A FAST  rock song " → "a fast rock song" |
| **Spell Correction** | Correct common misspellings of musical terms | "synthwave" → "synthwave", "bossa noav" → "bossa nova" |
| **Abbreviation Expansion** | Expand common abbreviations | "EDM" → "Electronic Dance Music", "BPM" → "beats per minute" |
| **Profanity Filtering** | Remove or flag inappropriate content | (Content moderation) |
| **Length Validation** | Ensure prompt is within acceptable length (10-500 characters) | Reject if too short or truncate if too long |

### 4.3 Stage 2: Semantic Parsing with Codex

The preprocessed prompt is sent to GPT Codex with a carefully crafted system prompt that instructs it to extract musical entities.

**System Prompt for Codex:**

```
You are a music production expert and semantic parser. Your task is to analyze a natural language description of a desired piece of music and extract structured musical parameters.

For each input, identify and extract the following entities if present:
- Genre and sub-genre
- Tempo (BPM or descriptive: slow, medium, fast, very fast)
- Mood and emotional qualities
- Instrumentation (specific instruments or instrument families)
- Musical structure (verse-chorus, AABA, through-composed, etc.)
- Key signature and mode (if mentioned)
- Time signature (if mentioned)
- Energy level (calm, moderate, energetic, intense)
- Reference artists or songs (for style inference)

If an entity is not explicitly mentioned, infer it from context or mark as "unspecified".

Output your analysis as a JSON object.
```

**Example Input/Output:**

*Input Prompt:* "Create an upbeat 80s synthwave track with pulsing bass and dreamy pads, something like The Midnight"

*Codex Output:*

```json
{
  "genre": "Synthwave",
  "sub_genre": "Retrowave",
  "tempo": {
    "type": "descriptive",
    "value": "medium-fast",
    "estimated_bpm": 118
  },
  "mood": ["Upbeat", "Dreamy", "Nostalgic"],
  "instrumentation": [
    {"name": "Synthesizer", "role": "lead"},
    {"name": "Bass Synthesizer", "role": "bass", "descriptor": "pulsing"},
    {"name": "Pad Synthesizer", "role": "pad", "descriptor": "dreamy"},
    {"name": "Drum Machine", "role": "drums"}
  ],
  "structure": "unspecified",
  "key_signature": "unspecified",
  "time_signature": "4/4",
  "energy_level": "energetic",
  "reference_artists": ["The Midnight"],
  "inferred_characteristics": {
    "era": "1980s",
    "production_style": "Polished, reverb-heavy"
  }
}
```

### 4.4 Stage 3: Parameter Mapping

The extracted entities are mapped to MusicAI's internal parameter schema. This involves:

1. **Genre Mapping**: Map genre/sub-genre to a predefined genre taxonomy and corresponding embedding.
2. **Tempo Normalization**: Convert BPM to a normalized value (0.0 - 1.0) based on a defined range (e.g., 60-200 BPM).
3. **Mood Embedding**: Map mood descriptors to a multi-dimensional mood embedding using a pre-trained mood model.
4. **Instrumentation Encoding**: Encode instrumentation as a multi-hot vector or embedding.
5. **Structure Encoding**: Map structure descriptions to predefined structural templates.

**Parameter Mapping Table:**

| Extracted Entity | MusicAI Parameter | Mapping Logic |
|------------------|-------------------|---------------|
| `genre` | `genre_embedding` | Lookup in genre taxonomy, retrieve pre-computed embedding |
| `tempo.estimated_bpm` | `tempo_normalized` | `(bpm - 60) / (200 - 60)` clamped to [0, 1] |
| `mood` | `mood_embedding` | Average embeddings of individual mood descriptors |
| `instrumentation` | `instrumentation_embedding` | Multi-hot encoding + learned instrument embeddings |
| `structure` | `structure_embedding` | Lookup in structure taxonomy |
| `energy_level` | `energy_level` | Map to [0, 1]: calm=0.2, moderate=0.5, energetic=0.75, intense=0.95 |
| `reference_artists` | (Style transfer hint) | Used by Aurora LLM for style inference |

### 4.5 Stage 4: Validation and Defaulting

After mapping, the system validates all parameters and applies defaults for any unspecified values.

**Validation Rules:**

| Parameter | Validation Rule | Default Value |
|-----------|-----------------|---------------|
| `genre_embedding` | Must be a valid embedding vector | "Pop" embedding |
| `tempo_normalized` | Must be in [0, 1] | 0.5 (120 BPM) |
| `mood_embedding` | Must be a valid embedding vector | "Neutral" embedding |
| `instrumentation_embedding` | At least one instrument | ["Piano"] |
| `structure_embedding` | Must be a valid structure | "Verse-Chorus" |
| `energy_level` | Must be in [0, 1] | 0.5 |

### 4.6 Stage 5: Embedding Generation

The validated parameters are combined into a single **Unified Control Embedding** vector that is passed to the MusicAI Control Encoder.

**Embedding Composition:**

```
unified_control_embedding = Concat(
    genre_embedding,          # dim: 128
    tempo_embedding,          # dim: 32 (learned from normalized tempo)
    mood_embedding,           # dim: 128
    instrumentation_embedding,# dim: 256
    structure_embedding,      # dim: 64
    energy_embedding          # dim: 32 (learned from energy level)
)
# Total dimension: 640
```

The Control Encoder then processes this unified embedding along with any additional Aurora LLM guidance to produce the final `control_embedding` tensor for the Structural Transformer.

---

## 5. Integration Points: Codex, Aurora LLM, and Control Encoder

This section specifies the integration points and data exchange protocols between the three key semantic processing components: GPT Codex, Aurora LLM, and the Control Encoder.

### 5.1 Integration Architecture

The three components form a **cascaded semantic processing pipeline**, where each component adds a layer of understanding and refinement.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEMANTIC PROCESSING PIPELINE                         │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │  GPT CODEX   │─────▶│  AURORA LLM  │─────▶│   CONTROL    │               │
│  │  (External)  │      │  (Local/LM   │      │   ENCODER    │               │
│  │              │      │   Studio)    │      │              │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│        │                      │                      │                       │
│        │                      │                      │                       │
│        ▼                      ▼                      ▼                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │  Structured  │      │  Semantic    │      │  Unified     │               │
│  │  Parameters  │      │  Guidance    │      │  Control     │               │
│  │  (JSON)      │      │  (Embedding) │      │  Embedding   │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Codex → Aurora LLM Integration

**Purpose**: Codex provides structured parameters that Aurora LLM uses for deeper semantic deconstruction and structural guidance.

**Data Exchange Format:**

```json
{
  "source": "codex",
  "version": "1.0",
  "payload": {
    "enhanced_prompt": "string - The Codex-enhanced prompt",
    "structured_parameters": {
      "genre": "string",
      "tempo_bpm": "number",
      "mood": ["string"],
      "instrumentation": ["string"],
      "structure": "string",
      "energy_level": "number"
    },
    "reference_context": {
      "artists": ["string"],
      "songs": ["string"],
      "era": "string"
    },
    "codex_confidence": "number - Overall confidence score"
  }
}
```

**Aurora LLM Processing:**

Aurora LLM receives the Codex output and performs the following tasks:

1. **Semantic Deconstruction**: Breaks down the enhanced prompt into atomic musical concepts.
2. **Structural Guidance**: Generates a detailed structural plan (e.g., section timings, transitions).
3. **Style Inference**: Uses reference artists/songs to infer stylistic nuances.
4. **Knowledge Augmentation**: Adds musical knowledge (e.g., typical chord progressions for the genre).

**Aurora LLM Output:**

```json
{
  "source": "aurora_llm",
  "version": "1.0",
  "payload": {
    "semantic_atoms": [
      {"concept": "pulsing_bass", "weight": 0.9},
      {"concept": "dreamy_atmosphere", "weight": 0.85},
      {"concept": "80s_nostalgia", "weight": 0.8}
    ],
    "structural_plan": {
      "sections": [
        {"name": "intro", "duration_ratio": 0.1, "energy": 0.4},
        {"name": "verse_1", "duration_ratio": 0.2, "energy": 0.6},
        {"name": "chorus_1", "duration_ratio": 0.15, "energy": 0.85},
        {"name": "verse_2", "duration_ratio": 0.2, "energy": 0.65},
        {"name": "chorus_2", "duration_ratio": 0.15, "energy": 0.9},
        {"name": "outro", "duration_ratio": 0.2, "energy": 0.5}
      ],
      "transitions": ["fade_in", "build", "drop", "build", "drop", "fade_out"]
    },
    "style_embedding": [0.1, 0.2, ...],
    "chord_progression_hint": ["Am", "F", "C", "G"],
    "aurora_confidence": 0.88
  }
}
```

### 5.3 Aurora LLM → Control Encoder Integration

**Purpose**: Aurora LLM's semantic guidance is combined with Codex's structured parameters to produce the final Unified Control Embedding.

**Data Exchange Format:**

```json
{
  "codex_params": {
    "genre_embedding": [0.1, 0.2, ...],
    "tempo_normalized": 0.65,
    "mood_embedding": [0.3, 0.4, ...],
    "instrumentation_embedding": [0.5, 0.6, ...],
    "structure_embedding": [0.7, 0.8, ...],
    "energy_level": 0.75
  },
  "aurora_guidance": {
    "semantic_atoms_embedding": [0.1, 0.2, ...],
    "structural_plan_embedding": [0.3, 0.4, ...],
    "style_embedding": [0.5, 0.6, ...]
  }
}
```

**Control Encoder Processing:**

The Control Encoder uses a learned fusion network to combine the two sources:

```python
class ControlEncoder(nn.Module):
    def __init__(self, codex_dim=640, aurora_dim=512, output_dim=512):
        super().__init__()
        self.codex_proj = nn.Linear(codex_dim, 256)
        self.aurora_proj = nn.Linear(aurora_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, codex_embedding, aurora_embedding):
        codex_feat = self.codex_proj(codex_embedding)
        aurora_feat = self.aurora_proj(aurora_embedding)
        fused = torch.cat([codex_feat, aurora_feat], dim=-1)
        return self.fusion(fused)
```

### 5.4 Fallback Hierarchy

In case of component failures, the system implements a fallback hierarchy:

| Scenario | Fallback Strategy |
|----------|-------------------|
| Codex unavailable | Use Aurora LLM alone with a simpler prompt parsing |
| Aurora LLM unavailable | Use Codex output directly with default structural guidance |
| Both unavailable | Use rule-based parameter extraction and default embeddings |

### 5.5 Latency Considerations

To minimize latency, the system supports parallel processing where possible:

1. **Parallel Codex + Aurora**: For prompts that don't require Codex enhancement, Aurora can process the raw prompt in parallel.
2. **Caching**: Frequently used genre/mood embeddings are cached.
3. **Streaming**: Codex suggestions can be streamed to the UI while Aurora processes in the background.

---

## 6. Data Models and Schemas

This section defines the data models and database schemas for storing Codex-generated suggestions, user refinements, and generation history.

### 6.1 Database Schema Overview

The following tables are added to the MusicAI database to support the Codex integration:

```sql
-- Codex Generation Sessions
CREATE TABLE codex_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id INT NOT NULL,
    original_prompt TEXT NOT NULL,
    enhanced_prompt TEXT,
    codex_response JSON,
    aurora_response JSON,
    final_parameters JSON,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Codex Suggestions
CREATE TABLE codex_suggestions (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    suggestion_type ENUM('genre', 'instrumentation', 'mood', 'structure', 'tempo', 'completion') NOT NULL,
    original_value TEXT,
    suggested_value TEXT NOT NULL,
    confidence DECIMAL(3, 2) NOT NULL,
    reasoning TEXT,
    user_accepted BOOLEAN DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES codex_sessions(id)
);

-- User Refinements
CREATE TABLE user_refinements (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    parameter_name VARCHAR(64) NOT NULL,
    original_value TEXT,
    refined_value TEXT NOT NULL,
    refinement_source ENUM('manual', 'suggestion_accepted', 'slider', 'preset') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES codex_sessions(id)
);

-- Generation Results
CREATE TABLE generation_results (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    audio_url TEXT NOT NULL,
    audio_duration_seconds DECIMAL(10, 2) NOT NULL,
    sample_rate INT NOT NULL,
    output_format VARCHAR(16) NOT NULL,
    generation_time_ms INT NOT NULL,
    codex_time_ms INT,
    aurora_time_ms INT,
    engine_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES codex_sessions(id)
);

-- User Feedback
CREATE TABLE user_feedback (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    result_id VARCHAR(36),
    feedback_type ENUM('positive', 'negative', 'neutral') NOT NULL,
    prompt_quality_rating TINYINT,
    suggestion_relevance_rating TINYINT,
    audio_quality_rating TINYINT,
    comments TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES codex_sessions(id),
    FOREIGN KEY (result_id) REFERENCES generation_results(id)
);

-- Indexes for performance
CREATE INDEX idx_codex_sessions_user_id ON codex_sessions(user_id);
CREATE INDEX idx_codex_sessions_status ON codex_sessions(status);
CREATE INDEX idx_codex_suggestions_session_id ON codex_suggestions(session_id);
CREATE INDEX idx_user_refinements_session_id ON user_refinements(session_id);
CREATE INDEX idx_generation_results_session_id ON generation_results(session_id);
CREATE INDEX idx_user_feedback_session_id ON user_feedback(session_id);
```

### 6.2 TypeScript Type Definitions

```typescript
// codex.types.ts

export interface CodexSession {
  id: string;
  userId: number;
  originalPrompt: string;
  enhancedPrompt: string | null;
  codexResponse: CodexResponse | null;
  auroraResponse: AuroraResponse | null;
  finalParameters: MusicalParameters | null;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  createdAt: Date;
  updatedAt: Date;
}

export interface CodexResponse {
  enhancedPrompt: string;
  suggestions: CodexSuggestion[];
  structuredParameters: StructuredMusicalParameters;
  metadata: {
    modelVersion: string;
    processingTimeMs: number;
    tokensUsed: number;
  };
}

export interface CodexSuggestion {
  id: string;
  sessionId: string;
  type: 'genre' | 'instrumentation' | 'mood' | 'structure' | 'tempo' | 'completion';
  originalValue: string | null;
  suggestedValue: string;
  confidence: number;
  reasoning: string;
  userAccepted: boolean | null;
  createdAt: Date;
}

export interface AuroraResponse {
  semanticAtoms: Array<{ concept: string; weight: number }>;
  structuralPlan: {
    sections: Array<{
      name: string;
      durationRatio: number;
      energy: number;
    }>;
    transitions: string[];
  };
  styleEmbedding: number[];
  chordProgressionHint: string[];
  confidence: number;
}

export interface StructuredMusicalParameters {
  genre: string;
  subGenre: string | null;
  tempoBpm: number;
  mood: string[];
  instrumentation: Array<{
    name: string;
    role: string;
    descriptor?: string;
  }>;
  structure: string;
  keySignature: string | null;
  timeSignature: string | null;
  energyLevel: number;
}

export interface MusicalParameters {
  genreEmbedding: number[];
  tempoNormalized: number;
  moodEmbedding: number[];
  instrumentationEmbedding: number[];
  structureEmbedding: number[];
  energyLevel: number;
  valence: number;
  danceability: number;
}

export interface UserRefinement {
  id: string;
  sessionId: string;
  parameterName: string;
  originalValue: string | null;
  refinedValue: string;
  refinementSource: 'manual' | 'suggestion_accepted' | 'slider' | 'preset';
  createdAt: Date;
}

export interface GenerationResult {
  id: string;
  sessionId: string;
  audioUrl: string;
  audioDurationSeconds: number;
  sampleRate: number;
  outputFormat: 'wav' | 'mp3' | 'flac';
  generationTimeMs: number;
  codexTimeMs: number | null;
  auroraTimeMs: number | null;
  engineTimeMs: number | null;
  createdAt: Date;
}

export interface UserFeedback {
  id: string;
  sessionId: string;
  resultId: string | null;
  feedbackType: 'positive' | 'negative' | 'neutral';
  promptQualityRating: number | null;
  suggestionRelevanceRating: number | null;
  audioQualityRating: number | null;
  comments: string | null;
  createdAt: Date;
}
```

### 6.3 JSON Schema for API Validation

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "definitions": {
    "EnhanceRequest": {
      "type": "object",
      "required": ["prompt"],
      "properties": {
        "prompt": {
          "type": "string",
          "minLength": 10,
          "maxLength": 500
        },
        "context": {
          "type": "object",
          "properties": {
            "genre_hint": { "type": "string" },
            "mood_hint": { "type": "string" },
            "duration_hint": { "type": "number", "minimum": 5, "maximum": 300 }
          }
        },
        "enhancement_level": {
          "type": "string",
          "enum": ["minimal", "moderate", "aggressive"],
          "default": "moderate"
        }
      }
    },
    "StructuredMusicalParameters": {
      "type": "object",
      "required": ["genre", "tempoBpm", "mood", "instrumentation", "energyLevel"],
      "properties": {
        "genre": { "type": "string" },
        "subGenre": { "type": ["string", "null"] },
        "tempoBpm": { "type": "number", "minimum": 40, "maximum": 240 },
        "mood": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        },
        "instrumentation": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "role"],
            "properties": {
              "name": { "type": "string" },
              "role": { "type": "string" },
              "descriptor": { "type": "string" }
            }
          },
          "minItems": 1
        },
        "structure": { "type": "string" },
        "keySignature": { "type": ["string", "null"] },
        "timeSignature": { "type": ["string", "null"] },
        "energyLevel": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    }
  }
}
```

---

## 7. Real-Time Prompt Refinement Implementation

This section provides implementation guidelines for integrating real-time Codex-assisted prompt refinement into the MusicAI web UI.

### 7.1 UI Component Architecture

The prompt refinement UI consists of the following components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROMPT INPUT AREA                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  [Textarea: User types natural language prompt here...]               │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  [Autocomplete Suggestions Dropdown]                            │  │  │
│  │  │  - "...with a driving beat and soaring synths"                  │  │  │
│  │  │  - "...featuring electric guitar and drums"                     │  │  │
│  │  │  - "...in the style of Daft Punk"                               │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  [Enhance Prompt Button]                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        SUGGESTION PANEL                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Enhanced Prompt:                                                      │  │
│  │  "Create an upbeat 80s synthwave track at 118 BPM with pulsing bass,  │  │
│  │   dreamy pads, and a nostalgic atmosphere, inspired by The Midnight"  │  │
│  │                                                          [Use This]   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Suggestions:                                                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │ Genre: Synthwave│ │ Tempo: 118 BPM  │ │ Mood: Nostalgic │                │
│  │ [Accept] [Edit] │ │ [Accept] [Edit] │ │ [Accept] [Edit] │                │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                │
│  ┌─────────────────┐ ┌─────────────────┐                                    │
│  │ Instruments:    │ │ Structure:      │                                    │
│  │ Synth, Bass,    │ │ Verse-Chorus    │                                    │
│  │ Drums           │ │ [Accept] [Edit] │                                    │
│  │ [Accept] [Edit] │ └─────────────────┘                                    │
│  └─────────────────┘                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PARAMETER SLIDERS                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Tempo:    [====|====================] 118 BPM                        │  │
│  │  Energy:   [================|========] 0.75                           │  │
│  │  Valence:  [==========|==============] 0.60                           │  │
│  │  Danceability: [==============|======] 0.80                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  [Generate Music]                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 React Component Implementation

```tsx
// components/PromptRefinement.tsx

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { trpc } from '@/lib/trpc';
import { useDebounce } from '@/hooks/useDebounce';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Loader2, Sparkles, Check, X, Edit2 } from 'lucide-react';

interface Suggestion {
  type: string;
  originalValue: string | null;
  suggestedValue: string;
  confidence: number;
  reasoning: string;
}

interface StructuredParameters {
  genre: string;
  tempoBpm: number;
  mood: string[];
  instrumentation: string[];
  structure: string;
  energyLevel: number;
}

export function PromptRefinement() {
  const [prompt, setPrompt] = useState('');
  const [enhancedPrompt, setEnhancedPrompt] = useState('');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [parameters, setParameters] = useState<StructuredParameters | null>(null);
  const [autocompleteSuggestions, setAutocompleteSuggestions] = useState<string[]>([]);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  
  const debouncedPrompt = useDebounce(prompt, 300);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Real-time autocomplete suggestions
  const suggestMutation = trpc.codex.suggest.useMutation({
    onSuccess: (data) => {
      setAutocompleteSuggestions(data.suggestions.map(s => s.text));
      setShowAutocomplete(data.suggestions.length > 0);
    },
  });

  // Prompt enhancement
  const enhanceMutation = trpc.codex.enhance.useMutation({
    onSuccess: (data) => {
      setEnhancedPrompt(data.enhanced_prompt);
      setSuggestions(data.suggestions);
      setParameters(data.structured_parameters);
      setIsEnhancing(false);
    },
    onError: () => {
      setIsEnhancing(false);
    },
  });

  // Fetch autocomplete suggestions as user types
  useEffect(() => {
    if (debouncedPrompt.length > 10) {
      suggestMutation.mutate({
        partial_prompt: debouncedPrompt,
        suggestion_count: 3,
      });
    } else {
      setShowAutocomplete(false);
    }
  }, [debouncedPrompt]);

  const handleEnhance = useCallback(() => {
    if (prompt.length < 10) return;
    setIsEnhancing(true);
    enhanceMutation.mutate({
      prompt,
      enhancement_level: 'moderate',
    });
  }, [prompt]);

  const handleAcceptSuggestion = useCallback((suggestion: Suggestion) => {
    // Update parameters based on accepted suggestion
    if (parameters) {
      const updated = { ...parameters };
      switch (suggestion.type) {
        case 'genre':
          updated.genre = suggestion.suggestedValue;
          break;
        case 'tempo':
          updated.tempoBpm = parseInt(suggestion.suggestedValue);
          break;
        case 'mood':
          updated.mood = [suggestion.suggestedValue];
          break;
        // ... handle other types
      }
      setParameters(updated);
    }
  }, [parameters]);

  const handleAutocompleteSelect = useCallback((suggestion: string) => {
    setPrompt(prev => prev + suggestion);
    setShowAutocomplete(false);
    textareaRef.current?.focus();
  }, []);

  const handleParameterChange = useCallback((param: string, value: number) => {
    if (parameters) {
      setParameters({
        ...parameters,
        [param]: value,
      });
    }
  }, [parameters]);

  return (
    <div className="space-y-6">
      {/* Prompt Input Area */}
      <Card className="p-4">
        <div className="relative">
          <Textarea
            ref={textareaRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the music you want to create... (e.g., 'An upbeat 80s synthwave track with pulsing bass and dreamy pads')"
            className="min-h-[120px] resize-none"
          />
          
          {/* Autocomplete Dropdown */}
          {showAutocomplete && (
            <div className="absolute left-0 right-0 top-full mt-1 bg-popover border rounded-md shadow-lg z-10">
              {autocompleteSuggestions.map((suggestion, index) => (
                <button
                  key={index}
                  className="w-full px-4 py-2 text-left hover:bg-accent text-sm"
                  onClick={() => handleAutocompleteSelect(suggestion)}
                >
                  <span className="text-muted-foreground">...{suggestion}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        
        <div className="mt-4 flex justify-end">
          <Button onClick={handleEnhance} disabled={isEnhancing || prompt.length < 10}>
            {isEnhancing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Enhancing...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Enhance Prompt
              </>
            )}
          </Button>
        </div>
      </Card>

      {/* Suggestion Panel */}
      {enhancedPrompt && (
        <Card className="p-4">
          <h3 className="font-semibold mb-2">Enhanced Prompt</h3>
          <p className="text-sm text-muted-foreground bg-muted p-3 rounded-md">
            {enhancedPrompt}
          </p>
          <Button variant="outline" size="sm" className="mt-2">
            Use This Prompt
          </Button>
          
          <div className="mt-4">
            <h4 className="font-medium mb-2">Suggestions</h4>
            <div className="flex flex-wrap gap-2">
              {suggestions.map((suggestion, index) => (
                <Badge
                  key={index}
                  variant="secondary"
                  className="flex items-center gap-2 px-3 py-1"
                >
                  <span className="text-xs text-muted-foreground">{suggestion.type}:</span>
                  <span>{suggestion.suggestedValue}</span>
                  <button
                    onClick={() => handleAcceptSuggestion(suggestion)}
                    className="ml-1 hover:text-green-500"
                  >
                    <Check className="h-3 w-3" />
                  </button>
                  <button className="hover:text-red-500">
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Parameter Sliders */}
      {parameters && (
        <Card className="p-4">
          <h3 className="font-semibold mb-4">Fine-tune Parameters</h3>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">
                Tempo: {parameters.tempoBpm} BPM
              </label>
              <Slider
                value={[parameters.tempoBpm]}
                min={60}
                max={200}
                step={1}
                onValueChange={([value]) => handleParameterChange('tempoBpm', value)}
                className="mt-2"
              />
            </div>
            <div>
              <label className="text-sm font-medium">
                Energy Level: {parameters.energyLevel.toFixed(2)}
              </label>
              <Slider
                value={[parameters.energyLevel * 100]}
                min={0}
                max={100}
                step={1}
                onValueChange={([value]) => handleParameterChange('energyLevel', value / 100)}
                className="mt-2"
              />
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
```

### 7.3 Debouncing and Throttling Strategy

To optimize API calls and provide a smooth user experience, the following debouncing and throttling strategies are implemented:

| Action | Strategy | Delay | Rationale |
|--------|----------|-------|-----------|
| Autocomplete suggestions | Debounce | 300ms | Avoid excessive API calls while typing |
| Prompt enhancement | No debounce | Immediate on button click | User-initiated action |
| Parameter slider changes | Debounce | 150ms | Smooth UI updates without excessive re-renders |
| Feedback submission | Throttle | 5s | Prevent accidental double submissions |

### 7.4 WebSocket Integration for Streaming Suggestions

For a more responsive experience, Codex suggestions can be streamed via WebSocket:

```typescript
// hooks/useCodexStream.ts

import { useEffect, useState, useCallback } from 'react';

interface StreamedSuggestion {
  type: 'partial' | 'complete';
  content: string;
}

export function useCodexStream(prompt: string) {
  const [streamedContent, setStreamedContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  const startStream = useCallback(() => {
    if (!prompt || prompt.length < 10) return;
    
    setIsStreaming(true);
    setStreamedContent('');

    const ws = new WebSocket(`${process.env.VITE_WS_URL}/codex/stream`);

    ws.onopen = () => {
      ws.send(JSON.stringify({ prompt }));
    };

    ws.onmessage = (event) => {
      const data: StreamedSuggestion = JSON.parse(event.data);
      if (data.type === 'partial') {
        setStreamedContent(prev => prev + data.content);
      } else if (data.type === 'complete') {
        setIsStreaming(false);
        ws.close();
      }
    };

    ws.onerror = () => {
      setIsStreaming(false);
    };

    ws.onclose = () => {
      setIsStreaming(false);
    };

    return () => {
      ws.close();
    };
  }, [prompt]);

  return { streamedContent, isStreaming, startStream };
}
```

---

## 8. Error Handling and Fallback Strategies

This section documents comprehensive error handling and fallback strategies to ensure system resilience when the Codex API is unavailable or returns errors.

### 8.1 Error Classification

Errors are classified into the following categories:

| Category | Error Codes | Description | User Impact |
|----------|-------------|-------------|-------------|
| **Transient** | `CODEX_TIMEOUT`, `CODEX_RATE_LIMITED`, `NETWORK_ERROR` | Temporary issues that may resolve on retry | Minor delay |
| **Recoverable** | `CODEX_UNAVAILABLE`, `CODEX_OVERLOADED` | Service temporarily down, fallback available | Degraded experience |
| **Permanent** | `INVALID_PROMPT`, `CONTENT_POLICY_VIOLATION` | User input issue, cannot be retried | User must modify input |
| **Critical** | `AUTHENTICATION_FAILED`, `QUOTA_EXCEEDED` | System configuration issue | Feature unavailable |

### 8.2 Retry Strategy

For transient errors, the system implements an exponential backoff retry strategy:

```typescript
// utils/retryWithBackoff.ts

interface RetryConfig {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  retryableErrors: string[];
}

const DEFAULT_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
  retryableErrors: ['CODEX_TIMEOUT', 'CODEX_RATE_LIMITED', 'NETWORK_ERROR'],
};

export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const { maxRetries, baseDelayMs, maxDelayMs, retryableErrors } = {
    ...DEFAULT_CONFIG,
    ...config,
  };

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      
      const errorCode = error.code || error.message;
      const isRetryable = retryableErrors.some(code => errorCode.includes(code));
      
      if (!isRetryable || attempt === maxRetries) {
        throw error;
      }

      const delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
      const jitter = Math.random() * 0.3 * delay; // Add 0-30% jitter
      await new Promise(resolve => setTimeout(resolve, delay + jitter));
    }
  }

  throw lastError;
}
```

### 8.3 Fallback Hierarchy

When Codex is unavailable, the system falls back to alternative processing methods:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FALLBACK HIERARCHY                                   │
│                                                                              │
│  Level 1: GPT Codex (Primary)                                               │
│     │                                                                        │
│     │ [Unavailable]                                                          │
│     ▼                                                                        │
│  Level 2: Aurora LLM (Secondary)                                            │
│     │  - Uses simplified prompt parsing                                      │
│     │  - Reduced suggestion quality                                          │
│     │                                                                        │
│     │ [Unavailable]                                                          │
│     ▼                                                                        │
│  Level 3: Rule-Based Parser (Tertiary)                                      │
│     │  - Keyword extraction                                                  │
│     │  - Predefined genre/mood mappings                                      │
│     │  - No intelligent suggestions                                          │
│     │                                                                        │
│     │ [Unavailable]                                                          │
│     ▼                                                                        │
│  Level 4: Default Parameters (Emergency)                                    │
│       - Uses sensible defaults                                               │
│       - User can manually adjust all parameters                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Fallback Implementation

```typescript
// services/promptProcessor.ts

import { CodexService } from './codexService';
import { AuroraService } from './auroraService';
import { RuleBasedParser } from './ruleBasedParser';
import { DEFAULT_PARAMETERS } from './defaults';

export class PromptProcessor {
  private codex: CodexService;
  private aurora: AuroraService;
  private ruleParser: RuleBasedParser;

  constructor() {
    this.codex = new CodexService();
    this.aurora = new AuroraService();
    this.ruleParser = new RuleBasedParser();
  }

  async processPrompt(prompt: string): Promise<ProcessingResult> {
    // Level 1: Try Codex
    try {
      const codexResult = await this.codex.enhance(prompt);
      return {
        source: 'codex',
        ...codexResult,
        fallbackUsed: false,
      };
    } catch (codexError) {
      console.warn('Codex unavailable, falling back to Aurora LLM', codexError);
    }

    // Level 2: Try Aurora LLM
    try {
      const auroraResult = await this.aurora.parsePrompt(prompt);
      return {
        source: 'aurora',
        ...auroraResult,
        fallbackUsed: true,
        fallbackReason: 'Codex unavailable',
      };
    } catch (auroraError) {
      console.warn('Aurora LLM unavailable, falling back to rule-based parser', auroraError);
    }

    // Level 3: Try Rule-Based Parser
    try {
      const ruleResult = this.ruleParser.parse(prompt);
      return {
        source: 'rule_based',
        ...ruleResult,
        fallbackUsed: true,
        fallbackReason: 'AI services unavailable',
      };
    } catch (ruleError) {
      console.warn('Rule-based parser failed, using defaults', ruleError);
    }

    // Level 4: Return Defaults
    return {
      source: 'defaults',
      enhancedPrompt: prompt,
      suggestions: [],
      structuredParameters: DEFAULT_PARAMETERS,
      fallbackUsed: true,
      fallbackReason: 'All processing methods failed',
    };
  }
}
```

### 8.5 Rule-Based Parser Implementation

```typescript
// services/ruleBasedParser.ts

const GENRE_KEYWORDS: Record<string, string> = {
  'rock': 'Rock',
  'pop': 'Pop',
  'jazz': 'Jazz',
  'classical': 'Classical',
  'electronic': 'Electronic',
  'edm': 'Electronic Dance Music',
  'hip hop': 'Hip Hop',
  'hip-hop': 'Hip Hop',
  'r&b': 'R&B',
  'country': 'Country',
  'metal': 'Metal',
  'synthwave': 'Synthwave',
  'ambient': 'Ambient',
  'folk': 'Folk',
  'blues': 'Blues',
  'reggae': 'Reggae',
  'soul': 'Soul',
  'funk': 'Funk',
};

const MOOD_KEYWORDS: Record<string, string> = {
  'happy': 'Happy',
  'sad': 'Sad',
  'energetic': 'Energetic',
  'calm': 'Calm',
  'relaxing': 'Relaxing',
  'upbeat': 'Upbeat',
  'melancholic': 'Melancholic',
  'aggressive': 'Aggressive',
  'romantic': 'Romantic',
  'nostalgic': 'Nostalgic',
  'dreamy': 'Dreamy',
  'dark': 'Dark',
  'bright': 'Bright',
  'intense': 'Intense',
  'peaceful': 'Peaceful',
};

const TEMPO_KEYWORDS: Record<string, number> = {
  'very slow': 60,
  'slow': 80,
  'moderate': 100,
  'medium': 110,
  'fast': 130,
  'very fast': 160,
  'upbeat': 120,
  'driving': 140,
};

export class RuleBasedParser {
  parse(prompt: string): ParseResult {
    const lowerPrompt = prompt.toLowerCase();
    
    // Extract genre
    let genre = 'Pop'; // Default
    for (const [keyword, genreName] of Object.entries(GENRE_KEYWORDS)) {
      if (lowerPrompt.includes(keyword)) {
        genre = genreName;
        break;
      }
    }

    // Extract mood
    const moods: string[] = [];
    for (const [keyword, moodName] of Object.entries(MOOD_KEYWORDS)) {
      if (lowerPrompt.includes(keyword)) {
        moods.push(moodName);
      }
    }
    if (moods.length === 0) moods.push('Neutral');

    // Extract tempo
    let tempoBpm = 120; // Default
    for (const [keyword, bpm] of Object.entries(TEMPO_KEYWORDS)) {
      if (lowerPrompt.includes(keyword)) {
        tempoBpm = bpm;
        break;
      }
    }

    // Extract BPM if explicitly mentioned
    const bpmMatch = lowerPrompt.match(/(\d{2,3})\s*bpm/);
    if (bpmMatch) {
      tempoBpm = parseInt(bpmMatch[1]);
    }

    // Calculate energy level from mood
    const energeticMoods = ['Energetic', 'Upbeat', 'Aggressive', 'Intense'];
    const calmMoods = ['Calm', 'Relaxing', 'Peaceful', 'Dreamy'];
    let energyLevel = 0.5;
    if (moods.some(m => energeticMoods.includes(m))) energyLevel = 0.8;
    if (moods.some(m => calmMoods.includes(m))) energyLevel = 0.3;

    return {
      enhancedPrompt: prompt, // No enhancement in rule-based mode
      suggestions: [], // No suggestions in rule-based mode
      structuredParameters: {
        genre,
        subGenre: null,
        tempoBpm,
        mood: moods,
        instrumentation: this.inferInstrumentation(genre),
        structure: 'Verse-Chorus',
        keySignature: null,
        timeSignature: '4/4',
        energyLevel,
      },
    };
  }

  private inferInstrumentation(genre: string): string[] {
    const genreInstruments: Record<string, string[]> = {
      'Rock': ['Electric Guitar', 'Bass', 'Drums'],
      'Pop': ['Synthesizer', 'Drums', 'Bass', 'Vocals'],
      'Jazz': ['Piano', 'Double Bass', 'Drums', 'Saxophone'],
      'Classical': ['Strings', 'Piano', 'Woodwinds'],
      'Electronic': ['Synthesizer', 'Drum Machine', 'Bass'],
      'Hip Hop': ['Drums', 'Bass', 'Synthesizer', 'Sampler'],
      'Synthwave': ['Synthesizer', 'Drum Machine', 'Bass Synthesizer'],
      'Ambient': ['Synthesizer', 'Pad', 'Field Recordings'],
    };
    return genreInstruments[genre] || ['Piano', 'Drums', 'Bass'];
  }
}
```

### 8.6 User Notification Strategy

When fallbacks are used, the system notifies the user appropriately:

| Fallback Level | User Notification | UI Indicator |
|----------------|-------------------|--------------|
| Aurora LLM | "Using alternative AI for prompt enhancement. Some suggestions may be limited." | Yellow warning badge |
| Rule-Based | "AI services temporarily unavailable. Using basic prompt analysis." | Orange warning banner |
| Defaults | "Unable to analyze prompt. Please adjust parameters manually." | Red error banner with manual controls |

---

## 9. Performance Optimization Strategies

This section specifies strategies for optimizing Codex API calls to minimize latency and improve user experience.

### 9.1 Latency Budget

The target latency budget for the full generation pipeline is:

| Stage | Target Latency | Max Acceptable |
|-------|----------------|----------------|
| Codex Enhancement | 500ms | 2000ms |
| Aurora LLM Processing | 300ms | 1000ms |
| Control Encoder | 50ms | 100ms |
| Tri-Hybrid Engine | 5000ms | 15000ms |
| **Total** | **5850ms** | **18100ms** |

### 9.2 Caching Strategy

Multiple caching layers are implemented to reduce redundant API calls:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CACHING LAYERS                                     │
│                                                                              │
│  Layer 1: Client-Side Cache (Browser)                                       │
│     - Recent autocomplete suggestions                                        │
│     - User's recent prompts and enhancements                                 │
│     - TTL: Session duration                                                  │
│                                                                              │
│  Layer 2: Application Cache (Redis)                                         │
│     - Prompt enhancement results (hash-based key)                            │
│     - Genre/mood embeddings                                                  │
│     - TTL: 1 hour for enhancements, 24 hours for embeddings                 │
│                                                                              │
│  Layer 3: Persistent Cache (Database)                                       │
│     - Historical prompt-parameter mappings                                   │
│     - User preference profiles                                               │
│     - TTL: Indefinite (with periodic cleanup)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Cache Implementation

```typescript
// services/cacheService.ts

import Redis from 'ioredis';
import crypto from 'crypto';

const redis = new Redis(process.env.REDIS_URL);

const CACHE_TTL = {
  ENHANCEMENT: 3600,      // 1 hour
  EMBEDDING: 86400,       // 24 hours
  AUTOCOMPLETE: 300,      // 5 minutes
};

export class CacheService {
  private generateKey(prefix: string, data: string): string {
    const hash = crypto.createHash('sha256').update(data).digest('hex').slice(0, 16);
    return `musicai:${prefix}:${hash}`;
  }

  async getEnhancement(prompt: string): Promise<EnhancementResult | null> {
    const key = this.generateKey('enhancement', prompt.toLowerCase().trim());
    const cached = await redis.get(key);
    return cached ? JSON.parse(cached) : null;
  }

  async setEnhancement(prompt: string, result: EnhancementResult): Promise<void> {
    const key = this.generateKey('enhancement', prompt.toLowerCase().trim());
    await redis.setex(key, CACHE_TTL.ENHANCEMENT, JSON.stringify(result));
  }

  async getEmbedding(type: string, value: string): Promise<number[] | null> {
    const key = this.generateKey(`embedding:${type}`, value.toLowerCase());
    const cached = await redis.get(key);
    return cached ? JSON.parse(cached) : null;
  }

  async setEmbedding(type: string, value: string, embedding: number[]): Promise<void> {
    const key = this.generateKey(`embedding:${type}`, value.toLowerCase());
    await redis.setex(key, CACHE_TTL.EMBEDDING, JSON.stringify(embedding));
  }

  async getAutocomplete(partialPrompt: string): Promise<string[] | null> {
    const key = this.generateKey('autocomplete', partialPrompt.toLowerCase().trim());
    const cached = await redis.get(key);
    return cached ? JSON.parse(cached) : null;
  }

  async setAutocomplete(partialPrompt: string, suggestions: string[]): Promise<void> {
    const key = this.generateKey('autocomplete', partialPrompt.toLowerCase().trim());
    await redis.setex(key, CACHE_TTL.AUTOCOMPLETE, JSON.stringify(suggestions));
  }
}
```

### 9.4 Request Batching

For scenarios where multiple parameters need to be processed, requests are batched:

```typescript
// services/batchProcessor.ts

interface BatchRequest {
  id: string;
  type: 'genre' | 'mood' | 'instrumentation';
  value: string;
  resolve: (result: number[]) => void;
  reject: (error: Error) => void;
}

export class BatchProcessor {
  private queue: BatchRequest[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;
  private readonly BATCH_SIZE = 10;
  private readonly BATCH_DELAY_MS = 50;

  async getEmbedding(type: string, value: string): Promise<number[]> {
    return new Promise((resolve, reject) => {
      this.queue.push({
        id: `${type}:${value}`,
        type: type as any,
        value,
        resolve,
        reject,
      });

      if (this.queue.length >= this.BATCH_SIZE) {
        this.processBatch();
      } else if (!this.batchTimeout) {
        this.batchTimeout = setTimeout(() => this.processBatch(), this.BATCH_DELAY_MS);
      }
    });
  }

  private async processBatch(): Promise<void> {
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }

    const batch = this.queue.splice(0, this.BATCH_SIZE);
    if (batch.length === 0) return;

    try {
      // Single API call for multiple embeddings
      const response = await fetch('/api/v1/embeddings/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requests: batch.map(r => ({ type: r.type, value: r.value })),
        }),
      });

      const results = await response.json();

      batch.forEach((request, index) => {
        if (results[index].error) {
          request.reject(new Error(results[index].error));
        } else {
          request.resolve(results[index].embedding);
        }
      });
    } catch (error) {
      batch.forEach(request => request.reject(error as Error));
    }
  }
}
```

### 9.5 Parallel Processing

Where possible, independent operations are executed in parallel:

```typescript
// services/parallelProcessor.ts

export async function processPromptParallel(prompt: string): Promise<ProcessingResult> {
  // Start Codex and Aurora in parallel for prompts that don't require enhancement
  const [codexResult, auroraResult] = await Promise.allSettled([
    codexService.enhance(prompt),
    auroraService.parsePrompt(prompt),
  ]);

  // Use Codex result if available, otherwise fall back to Aurora
  if (codexResult.status === 'fulfilled') {
    return {
      ...codexResult.value,
      auroraGuidance: auroraResult.status === 'fulfilled' ? auroraResult.value : null,
    };
  } else if (auroraResult.status === 'fulfilled') {
    return {
      ...auroraResult.value,
      source: 'aurora',
      fallbackUsed: true,
    };
  }

  throw new Error('Both Codex and Aurora failed');
}
```

### 9.6 Connection Pooling

HTTP connection pooling is used to reduce connection overhead:

```typescript
// services/httpClient.ts

import axios from 'axios';
import https from 'https';

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10,
  timeout: 30000,
});

export const codexClient = axios.create({
  baseURL: process.env.CODEX_API_URL,
  timeout: 10000,
  httpsAgent,
  headers: {
    'Authorization': `Bearer ${process.env.CODEX_API_KEY}`,
    'Content-Type': 'application/json',
  },
});

// Add response time logging
codexClient.interceptors.response.use(
  (response) => {
    const duration = Date.now() - (response.config as any).startTime;
    console.log(`Codex API call completed in ${duration}ms`);
    return response;
  },
  (error) => {
    const duration = Date.now() - (error.config as any).startTime;
    console.error(`Codex API call failed after ${duration}ms`, error.message);
    return Promise.reject(error);
  }
);

codexClient.interceptors.request.use((config) => {
  (config as any).startTime = Date.now();
  return config;
});
```

### 9.7 Precomputation and Warm-Up

Common embeddings and model weights are precomputed and loaded at startup:

```typescript
// services/warmUp.ts

export async function warmUpServices(): Promise<void> {
  console.log('Warming up services...');

  // Preload common genre embeddings
  const commonGenres = ['Pop', 'Rock', 'Electronic', 'Jazz', 'Classical', 'Hip Hop'];
  await Promise.all(
    commonGenres.map(genre => embeddingService.preloadGenreEmbedding(genre))
  );

  // Preload common mood embeddings
  const commonMoods = ['Happy', 'Sad', 'Energetic', 'Calm', 'Romantic'];
  await Promise.all(
    commonMoods.map(mood => embeddingService.preloadMoodEmbedding(mood))
  );

  // Warm up Codex connection
  await codexClient.post('/health');

  // Warm up Aurora LLM connection
  await auroraService.ping();

  console.log('Services warmed up successfully');
}
```

### 9.8 Performance Monitoring

Key performance metrics are tracked and monitored:

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Codex API P50 Latency | < 500ms | > 1000ms |
| Codex API P99 Latency | < 2000ms | > 5000ms |
| Cache Hit Rate | > 60% | < 40% |
| Fallback Rate | < 5% | > 15% |
| Error Rate | < 1% | > 5% |

---

## 10. Appendices

### Appendix A: Codex System Prompt Template

```
You are a music production expert and semantic parser integrated into the MusicAI system. Your task is to analyze natural language descriptions of desired music and extract structured musical parameters.

## Your Capabilities:
1. Parse complex musical descriptions
2. Infer missing parameters from context
3. Suggest enhancements to improve musical coherence
4. Map descriptions to structured parameter schemas

## Output Format:
Always respond with valid JSON matching the following schema:
{
  "enhanced_prompt": "string",
  "suggestions": [...],
  "structured_parameters": {...}
}

## Guidelines:
- Be specific about tempo (provide BPM estimates)
- Infer instrumentation from genre if not specified
- Consider musical coherence when suggesting parameters
- Provide confidence scores for inferred values
- Include reasoning for suggestions

## Example:
Input: "A chill lo-fi beat for studying"
Output: {
  "enhanced_prompt": "A chill lo-fi hip hop beat at 85 BPM with warm piano chords, vinyl crackle, and soft drums, perfect for studying and relaxation",
  "suggestions": [
    {"type": "tempo", "suggestedValue": "85", "confidence": 0.85, "reasoning": "Lo-fi beats typically range from 70-90 BPM"}
  ],
  "structured_parameters": {
    "genre": "Lo-Fi Hip Hop",
    "tempo_bpm": 85,
    "mood": ["Chill", "Relaxing", "Focused"],
    "instrumentation": ["Piano", "Drums", "Vinyl Crackle", "Bass"],
    "energy_level": 0.3
  }
}
```

### Appendix B: Error Code Reference

| Error Code | HTTP Status | Description | User Message |
|------------|-------------|-------------|--------------|
| `CODEX_TIMEOUT` | 504 | Codex API request timed out | "Request timed out. Please try again." |
| `CODEX_RATE_LIMITED` | 429 | Rate limit exceeded | "Too many requests. Please wait a moment." |
| `CODEX_UNAVAILABLE` | 503 | Codex service unavailable | "AI service temporarily unavailable." |
| `CODEX_OVERLOADED` | 503 | Codex service overloaded | "Service is busy. Using alternative processing." |
| `INVALID_PROMPT` | 400 | Prompt validation failed | "Please provide a more detailed description." |
| `CONTENT_POLICY_VIOLATION` | 400 | Content policy violation | "Your prompt contains disallowed content." |
| `AUTHENTICATION_FAILED` | 401 | API authentication failed | "Service configuration error. Please contact support." |
| `QUOTA_EXCEEDED` | 403 | API quota exceeded | "Service quota exceeded. Please try again later." |
| `NETWORK_ERROR` | 502 | Network connectivity issue | "Network error. Please check your connection." |

### Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Codex** | OpenAI's GPT-based model optimized for code and structured output generation |
| **Aurora LLM** | MusicAI's local LLM for semantic deconstruction and musical knowledge |
| **Control Encoder** | Neural network that combines semantic information into a unified embedding |
| **RAVE** | Realtime Audio Variational autoEncoder for high-fidelity audio synthesis |
| **DDSP** | Differentiable Digital Signal Processing for interpretable audio control |
| **Unified Control Embedding** | Combined vector representation of all musical parameters |
| **Structural Transformer** | Transformer model that generates RAVE latent vectors |
| **Ultimate Mixer** | Neural network that blends RAVE and DDSP outputs |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-07 | Manus AI | Initial release |

---

**End of Document**
