# Kokoro-82M TTS Implementations

This directory contains implementations using the Kokoro-82M text-to-speech model.

## Files Overview

1. **podcast_generator_kokoro.py**
   - Standard implementation for local use
   - Supports environment variables or AWS CLI credentials
   - Falls back to demo dialogue if Bedrock unavailable

2. **podcast_generator_kokoro_ec2.py** ⭐ **Production Ready**
   - Optimized for EC2 deployment
   - Automatic IAM role authentication
   - No fallback mode - expects Bedrock to work
   - GPU acceleration support

## Key Advantages
- **No token limits** - Handles entire dialogues without chunking
- **54 built-in voices** - Natural sounding options
- **Faster generation** - 82M vs 1.6B parameters
- **Simpler implementation** - No complex chunking logic

## AWS Bedrock Integration

Both implementations use AWS Bedrock with Claude 3.5 Sonnet for intelligent dialogue generation:

### Dialogue Generation Process
1. **Content Extraction**: Fetches and parses web articles
2. **AI Prompt**: Sends content to Claude 3.5 Sonnet via Bedrock
3. **Dialogue Creation**: Claude generates natural conversation between two speakers
4. **TTS Synthesis**: Kokoro converts dialogue to speech with selected voices

### Bedrock Configuration
```python
# Automatic with IAM roles on EC2
self.bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)
self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
```

### Required IAM Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:*:*:model/anthropic.claude-3-5-sonnet*"
        }
    ]
}
```

## Available Voices

### Recommended Voice Pairs
- **Natural**: af_nova (S1) + am_liam (S2)
- **Professional**: af_alloy (S1) + am_echo (S2)
- **Friendly**: af_bella (S1) + am_michael (S2)

### All Voices
- **American Female**: af_nova, af_alloy, af_bella, af_heart, af_jessica, af_nicole, af_sarah
- **American Male**: am_liam, am_echo, am_michael, am_adam, am_onyx
- **British Female**: bf_emma, bf_isabella, bf_alice, bf_lily
- **British Male**: bm_george, bm_lewis, bm_daniel

## Usage Examples

### Local Development
```bash
# With AWS credentials configured
python src/kokoro/podcast_generator_kokoro.py \
  --url https://example.com/article \
  --s1-voice af_nova \
  --s2-voice am_liam \
  --duration 5
```

### EC2 Production
```bash
# IAM role handles authentication automatically
python src/kokoro/podcast_generator_kokoro_ec2.py \
  --url https://example.com/article \
  --duration 10 \
  --speed 0.9
```

## Claude 3.5 Sonnet Features

The dialogue generation leverages Claude's advanced capabilities:
- **Contextual Understanding**: Analyzes article content deeply
- **Natural Conversation**: Creates realistic back-and-forth dialogue
- **Personality Distinction**: S1 is analytical, S2 is curious/reactive
- **Pacing Control**: Adds pauses and filler words for natural speech
- **Content Coverage**: Ensures all key points are discussed
- **Engaging Format**: Makes complex topics accessible