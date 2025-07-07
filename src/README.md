# Source Code

This directory contains the main implementations of the podcast generator.

## Structure

### `/dia` - Dia-1.6B TTS Implementations
- **podcast_generator_final.py** - Original implementation with voice cloning
- **podcast_generator_chunked.py** - Basic chunking approach (15-line chunks)
- **podcast_generator_chunked_fixed.py** - Improved chunking (10-line chunks)
- **podcast_generator_small_chunks.py** - Optimal chunking (4-line chunks) ⭐ Recommended
- **podcast_generator_fixed_trimming.py** - Advanced trimming with silence detection
- **podcast_generator_no_overlap.py** - Smart chunking without overlap
- **podcast_generator_consistent_voices.py** - Voice consistency improvements
- **podcast_generator_extended.py** - Extended features

### `/kokoro` - Kokoro-82M TTS Implementations
- **podcast_generator_kokoro.py** - Standard implementation with multiple voices
- **podcast_generator_kokoro_ec2.py** - EC2-optimized with IAM role support ⭐ Production

## Key Differences

### Dia-1.6B
- Supports voice cloning
- Requires chunking for long content
- Higher quality output
- More resource intensive

### Kokoro-82M
- No token limits
- 54 built-in voices
- Faster generation
- Lighter resource usage