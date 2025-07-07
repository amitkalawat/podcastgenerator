# Audio Samples

This directory contains sample outputs from both TTS models.

## Files

### Dia-1.6B Samples
- **dia_1.6b_sample.wav** (2.5 minutes, 148 seconds)
  - Generated using voice cloning from `example_prompt.mp3`
  - Uses small chunk processing (4-line chunks)
  - Complete dialogue with all sections included
  - Natural conversation about Claude 4

### Kokoro-82M Samples  
- **kokoro_82m_ec2_sample.wav** (3.9 minutes)
  - Generated on EC2 with GPU acceleration
  - Full AI dialogue from AWS Bedrock
  - af_nova (S1) + am_liam (S2)
  - Production example with IAM role auth

- **kokoro_82m_sample.wav** (4.1 minutes)
  - Full podcast with natural voices
  - af_nova (S1) + am_liam (S2)
  - No chunking required
  
- **kokoro_82m_demo.wav** (54 seconds)
  - Short demo with fallback dialogue
  - Shows voice quality and pacing

### Voice Cloning
- **example_prompt.mp3**
  - Sample audio for voice cloning with Dia-1.6B
  - Contains: "[S1] Open weights text to dialogue model. [S2] You get full control over scripts and voices."

## Generation Commands

```bash
# Dia-1.6B with voice cloning (small chunks)
python podcast_generator_small_chunks.py \
  --voice-clone-audio example_prompt.mp3 \
  --url https://www.anthropic.com/news/claude-4

# Kokoro-82M with natural voices  
python podcast_generator_kokoro.py \
  --s1-voice af_nova \
  --s2-voice am_liam \
  --duration 5

# Kokoro-82M on EC2 with Bedrock
python podcast_generator_kokoro_ec2.py \
  --url https://www.anthropic.com/news/claude-4 \
  --duration 3
```