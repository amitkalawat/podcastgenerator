# Dia-1.6B TTS Implementations

This directory contains various implementations using the Dia-1.6B text-to-speech model.

## Files Overview

### Core Implementations
1. **podcast_generator_final.py**
   - Original implementation
   - Basic voice cloning support
   - ~29 second limitation due to token limits

2. **podcast_generator_small_chunks.py** ⭐ **Recommended**
   - Optimal 4-line chunking strategy
   - Most reliable for complete dialogue
   - Best balance of quality and reliability

3. **podcast_generator_fixed_trimming.py**
   - Advanced silence detection for trimming
   - Prevents cut-off words at chunk beginnings
   - Best for production use

### Evolution of Chunking Strategies
- **podcast_generator_chunked.py** - Initial 15-line chunks (some content missing)
- **podcast_generator_chunked_fixed.py** - Improved 10-line chunks
- **podcast_generator_small_chunks.py** - Optimal 4-line chunks
- **podcast_generator_no_overlap.py** - Smart chunking at speaker changes

### Special Features
- **podcast_generator_consistent_voices.py** - Maintains voice consistency across chunks
- **podcast_generator_extended.py** - Extended features and improvements

## Usage Example

```bash
# Basic usage with voice cloning
python src/dia/podcast_generator_small_chunks.py \
  --voice-clone-audio samples/example_prompt.mp3 \
  --url https://example.com/article \
  --duration 5

# With custom parameters
python src/dia/podcast_generator_fixed_trimming.py \
  --voice-clone-audio samples/example_prompt.mp3 \
  --voice-clone-transcript "[S1] Custom voice. [S2] Another voice." \
  --speed 0.85 \
  --seed 42
```

## Key Features
- Voice cloning from audio samples
- Adjustable speech speed (0.8-1.0)
- Seed control for reproducibility
- AWS Bedrock integration for dialogue generation