# Kokoro-82M TTS Implementation Summary

## Overview
I've successfully created a new version of the podcast generator using Kokoro-82M, an open-weight TTS model that offers several advantages over Dia-1.6B.

## Key Differences from Dia-1.6B

### Advantages:
1. **No token limits**: Kokoro can handle longer text segments without chunking
2. **Smaller model**: 82M parameters vs 1.6B (faster, less memory)
3. **Multiple built-in voices**: 54 voices across different languages
4. **Simpler API**: Easier to use and maintain
5. **Active development**: Recent model with good documentation

### Limitations:
1. **No voice cloning**: Cannot clone custom voices from audio samples
2. **Fixed voices**: Limited to pre-trained voices
3. **Different quality**: Being smaller, audio quality differs from Dia-1.6B

## Implementation Details

### Installation:
```bash
pip install kokoro>=0.9.2 soundfile
# Also requires espeak-ng system package
```

### Key Changes:
1. **Voice Selection**: Using different pre-trained voices for S1 and S2
   - S1: `af_heart` (American Female)
   - S2: `am_adam` (American Male)

2. **Audio Generation**: Kokoro returns a generator that yields audio chunks
```python
generator = self.tts_pipeline(text, voice=voice)
for i, (gs, ps, audio_chunk) in enumerate(generator):
    audio_chunks.append(audio_chunk)
```

3. **Processing Flow**:
   - Split dialogue by speaker
   - Generate audio for each speaker segment with appropriate voice
   - Concatenate segments with small silence gaps

## Available Voices

### American English (lang_code='a'):
- **Female**: af_heart, af_alloy, af_bella, af_jessica, af_nicole, af_nova, af_sarah
- **Male**: am_adam, am_echo, am_liam, am_michael, am_onyx

### British English (lang_code='b'):
- **Female**: bf_alice, bf_emma, bf_isabella, bf_lily
- **Male**: bm_daniel, bm_george, bm_lewis

## Usage Example:
```bash
python podcast_generator_kokoro.py \
  --url https://example.com/article \
  --s1-voice af_heart \
  --s2-voice am_adam \
  --duration 5 \
  --speed 0.85
```

## Results
- Successfully generated a 54.7-second podcast using fallback dialogue
- Audio quality is good with distinct voices for each speaker
- No chunking issues or trimming problems
- Faster generation compared to Dia-1.6B

## Files Created:
1. `podcast_generator_kokoro.py` - Main implementation
2. `podcast_kokoro_test.wav` - Sample generated audio

The Kokoro implementation provides a simpler, faster alternative to Dia-1.6B, though without voice cloning capabilities.