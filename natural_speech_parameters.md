# Natural Speech Parameters for Dia-1.6B

## Key Adjustments for Slower, More Natural Speech

### 1. Speech Speed Control
- **New Parameter**: `--speed` (0.8 to 1.0)
- **Recommended**: 0.85 for natural pacing
- **Implementation**: Post-processing audio resampling

### 2. Generation Parameters Optimization

#### For Voice Cloning:
```python
cfg_scale=3.5,       # Lower from 4.0 for more natural flow
temperature=1.4,     # Lower from 1.8 for controlled output
top_p=0.92,         # Slightly higher from 0.90 for variety
cfg_filter_top_k=40, # Balanced from 50
```

#### For Standard Generation:
```python
cfg_scale=2.8,       # Lower from 3.0 for natural speech
temperature=1.4,     # Lower from 1.8 for consistency
top_p=0.92,         # Slightly higher for variety
cfg_filter_top_k=40, # Balanced filtering
```

### 3. Dialogue Prompt Improvements
- Added instructions for natural pauses using "..."
- Encouraged use of commas for breathing points
- Added filler words ("well", "you know", "I mean")
- Shorter sentence chunks for better pacing

### 4. Parameter Effects

| Parameter | Original | Adjusted | Effect |
|-----------|----------|----------|---------|
| cfg_scale | 3.0-4.0 | 2.8-3.5 | Lower = more natural flow, less rigid |
| temperature | 1.8 | 1.4 | Lower = more controlled, consistent |
| top_p | 0.90 | 0.92 | Higher = more variety in expression |
| speed | N/A | 0.85 | Slower playback for natural pacing |

### 5. Usage Examples

**Natural Voice Cloning**:
```bash
python podcast_dia_bedrock_v5.1_speed_control.py \
    --url "https://example.com/article" \
    --voice-clone-audio example_prompt.mp3 \
    --speed 0.85
```

**Slower Standard Generation**:
```bash
python podcast_dia_bedrock_v5.1_speed_control.py \
    --url "https://example.com/article" \
    --speed 0.8 \
    --seed 42
```

### 6. Additional Tips

1. **Dialogue Structure**:
   - Use "..." for natural pauses
   - Break long sentences into chunks
   - Add conversational fillers

2. **Speed Recommendations**:
   - 0.80: Very slow, clear enunciation
   - 0.85: Natural conversational pace (recommended)
   - 0.90: Slightly slower than default
   - 0.95: Near-normal speed
   - 1.00: Original speed (often too fast)

3. **Fine-tuning**:
   - Lower cfg_scale for more expressive speech
   - Lower temperature for more consistent tone
   - Adjust top_p for variety vs consistency balance

### 7. Example Dialogue with Natural Pacing

```
[S1] Have you heard about AWS's Project Rainier? It's... quite fascinating.
[S2] No, I haven't. What is it?
[S1] Well, they're building what might be... the world's most powerful AI training computer.
[S2] That sounds incredible! Tell me more.
```

The combination of adjusted parameters and speed control creates more natural, easier-to-follow podcast audio.