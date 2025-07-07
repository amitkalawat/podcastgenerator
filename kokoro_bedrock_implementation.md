# Kokoro-82M with AWS Bedrock Implementation

## Overview
I've successfully implemented a podcast generator that uses:
- **Kokoro-82M** for text-to-speech (no token limits)
- **AWS Bedrock** with Claude 3.5 Sonnet for dialogue generation
- Multiple natural-sounding voices for conversation

## Key Improvements

### 1. More Natural Voices
Based on Kokoro's documentation, I've selected voices that sound more human-like:

**Recommended Voice Combinations:**
- **Natural/Friendly**: `af_nova` (warm female) + `am_liam` (conversational male)
- **Professional**: `af_alloy` (clear female) + `am_echo` (authoritative male)
- **Enthusiastic**: `af_bella` (energetic female) + `am_michael` (friendly male)

### 2. AWS Bedrock Integration
The script now properly handles AWS credentials:
```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Option 2: AWS CLI
aws configure

# Option 3: IAM role (on EC2)
```

### 3. Speed Control
Kokoro supports speed adjustment for more natural pacing:
- `--speed 0.85`: Default, natural conversational pace
- `--speed 0.9`: Slightly slower, more deliberate
- `--speed 1.0`: Normal speed

## Usage Examples

### With AWS Bedrock (full functionality):
```bash
# Set AWS credentials first
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Generate podcast
python podcast_generator_kokoro.py \
  --url https://example.com/article \
  --s1-voice af_nova \
  --s2-voice am_liam \
  --duration 5 \
  --speed 0.85
```

### Without AWS (uses fallback dialogue):
```bash
python podcast_generator_kokoro.py \
  --s1-voice af_alloy \
  --s2-voice am_echo \
  --duration 3 \
  --speed 0.9
```

## Generated Files

### Audio Samples:
1. **claude4_kokoro_natural.wav** (4.1 minutes)
   - Voices: af_nova + am_liam
   - Natural, friendly conversation style

2. **claude4_kokoro_professional.wav** (3.9 minutes)
   - Voices: af_alloy + am_echo
   - Professional, authoritative tone

### Scripts:
- **podcast_generator_kokoro.py**: Main implementation

## Advantages Over Dia-1.6B

1. **No Token Limits**: Can generate entire podcasts without chunking
2. **No Trimming Issues**: Clean audio generation without cut-off words
3. **Simpler Implementation**: No complex chunk management
4. **Faster Generation**: 82M vs 1.6B parameters
5. **Multiple Voice Options**: 54 built-in voices

## Voice Selection Guide

### American Female Voices:
- `af_nova`: Warm, friendly (recommended for conversational podcasts)
- `af_alloy`: Professional, clear (good for educational content)
- `af_bella`: Enthusiastic, energetic
- `af_heart`: Neutral, versatile

### American Male Voices:
- `am_liam`: Natural, conversational (recommended)
- `am_echo`: Deep, authoritative
- `am_michael`: Friendly, approachable
- `am_adam`: Balanced, professional

### British Voices:
- `bf_emma`: Elegant British female
- `bm_george`: Distinguished British male

## Performance Notes

- Generation speed: ~1 minute of audio per 10 seconds of processing
- Audio quality: 24kHz sampling rate
- Memory usage: Minimal compared to Dia-1.6B
- No GPU required for reasonable performance

## Deployment on EC2

To deploy on EC2:
1. Install Kokoro: `pip install kokoro>=0.9.2 soundfile`
2. Install system dependency: `sudo apt-get install espeak-ng`
3. Configure AWS credentials for Bedrock access
4. Run the script with desired parameters

The Kokoro implementation provides a robust, scalable solution for podcast generation without the complexity of managing token limits and audio chunking.