# Podcast Generator

Generate natural-sounding conversational podcasts from any web content using AI dialogue generation and advanced text-to-speech models.

## 🌟 Features

### Two TTS Model Options

#### 1. **Dia-1.6B** (Advanced)
- 🔊 Voice cloning from audio samples
- 🎭 Consistent speaker identity
- 📊 1.6B parameters for high quality
- ⚡ Requires chunking for long content

#### 2. **Kokoro-82M** (Lightweight)
- 🚀 No token limits - handles long content
- 🎙️ 54 built-in natural voices
- 💨 Faster generation (82M parameters)
- 🔧 Simpler implementation

### Core Features
- 🤖 **AI Dialogue**: Claude 3.5 Sonnet via AWS Bedrock
- 🎚️ **Speed Control**: Adjustable speech pace (0.8-1.0x)
- 🌐 **Web Scraping**: Automatic content extraction
- ☁️ **EC2 Ready**: IAM role authentication support
- 🔒 **Secure**: No hardcoded credentials

## 🎧 Audio Samples

Listen to sample podcasts generated about Claude 4 announcement:

### Dia-1.6B Sample (with voice cloning)
[🔊 Listen: dia_1.6b_sample.wav](https://github.com/amitkalawat/podcastgenerator/raw/main/samples/dia_1.6b_sample.wav) (2.5 minutes, 12MB)
- ✨ Voice cloned from [example_prompt.mp3](samples/example_prompt.mp3)
- 🎯 Small chunk processing (4-line chunks) for complete dialogue
- 📊 148 seconds total, all dialogue sections included
- 🔧 Generated with: `podcast_generator_small_chunks.py`

### Kokoro-82M Samples
[🔊 Listen: kokoro_82m_ec2_sample.wav](https://github.com/amitkalawat/podcastgenerator/raw/main/samples/kokoro_82m_ec2_sample.wav) (3.9 minutes, 11MB)
- 🚀 Generated on EC2 with AWS Bedrock + IAM roles
- 🎭 Voices: af_nova (warm female) + am_liam (conversational male)
- 💎 Full AI-generated dialogue about Claude 4
- 🔧 Generated with: `podcast_generator_kokoro_ec2.py`

[🔊 Listen: kokoro_82m_sample.wav](https://github.com/amitkalawat/podcastgenerator/raw/main/samples/kokoro_82m_sample.wav) (4.1 minutes, 11MB)
- 🎯 Local generation example
- 🚀 No chunking required - handles full dialogue

[🔊 Listen: kokoro_82m_demo.wav](https://github.com/amitkalawat/podcastgenerator/raw/main/samples/kokoro_82m_demo.wav) (54 seconds, 2.5MB)
- 💨 Quick demo with fallback dialogue
- 🎙️ Shows voice quality and natural pacing

## 📁 Repository Structure

```
podcastgenerator/
├── src/                    # Source code
│   ├── dia/               # Dia-1.6B implementations
│   │   ├── podcast_generator_small_chunks.py     # ⭐ Recommended
│   │   ├── podcast_generator_fixed_trimming.py   # Advanced trimming
│   │   └── ...
│   └── kokoro/            # Kokoro-82M implementations
│       ├── podcast_generator_kokoro.py           # Local version
│       └── podcast_generator_kokoro_ec2.py       # ⭐ EC2 production
├── samples/               # Audio samples & demos
├── dialogs/              # Generated dialogue examples
├── scripts/              # Deployment scripts
├── tests/                # Test scripts
├── docs/                 # Documentation
└── requirements.txt      # Python dependencies
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- AWS account with Bedrock access
- 8GB+ RAM recommended

### Python Dependencies
```bash
pip install torch transformers scipy numpy beautifulsoup4 requests boto3 soundfile
pip install git+https://github.com/nari-labs/dia.git
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd podcastgenerator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install git+https://github.com/nari-labs/dia.git
```

3. Configure AWS credentials for Bedrock access:
```bash
aws configure
```

## 🚀 Quick Start

### Using Dia-1.6B (with voice cloning)
```bash
python src/dia/podcast_generator_small_chunks.py \
  --voice-clone-audio samples/example_prompt.mp3 \
  --url https://example.com/article
```

### Using Kokoro-82M (no token limits)
```bash
python src/kokoro/podcast_generator_kokoro.py \
  --url https://example.com/article \
  --s1-voice af_nova \
  --s2-voice am_liam
```

## Usage

### Basic Usage

Generate a podcast from a web article:
```bash
python src/dia/podcast_generator_final.py --url "https://example.com/article" --output my_podcast.wav
```

### With Voice Cloning

Clone voices from an audio sample:
```bash
python src/dia/podcast_generator_final.py \
    --url "https://example.com/article" \
    --voice-clone-audio "reference_audio.mp3" \
    --voice-clone-transcript "[S1] Reference text. [S2] More reference text." \
    --output cloned_podcast.wav
```

### Natural Speech Settings

For slower, more natural speech:
```bash
python src/dia/podcast_generator_final.py \
    --url "https://example.com/article" \
    --speed 0.85 \
    --output natural_podcast.wav
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--url` | URL of the article to convert | Required |
| `--output`, `-o` | Output audio file path | `podcast_output.wav` |
| `--speed` | Speech speed (0.8-1.0) | `0.85` |
| `--voice-clone-audio` | Audio file for voice cloning | None |
| `--voice-clone-transcript` | Transcript of voice clone audio | None |
| `--seed` | Random seed for reproducibility | None |
| `--cpu` | Force CPU usage | False |
| `--region` | AWS region for Bedrock | `us-west-2` |

## Examples

### 1. Standard Podcast Generation
```bash
python src/dia/podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --output aws_rainier_podcast.wav
```

### 2. Voice Cloning with Example Audio
```bash
python src/dia/podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --voice-clone-audio example_prompt.mp3 \
    --speed 0.85 \
    --output aws_rainier_cloned.wav
```

### 3. Reproducible Generation with Seed
```bash
python src/dia/podcast_generator_final.py \
    --url "https://example.com/article" \
    --seed 42 \
    --speed 0.9 \
    --output reproducible_podcast.wav
```

## Voice Cloning Guide

### Preparing Audio Samples

1. **Duration**: 5-10 seconds of clear speech
2. **Format**: MP3 or WAV
3. **Quality**: Clear audio without background noise
4. **Content**: Should include both speakers if cloning multiple voices

### Transcript Format

Use `[S1]` and `[S2]` tags for speakers:
```
[S1] This is the first speaker's text.
[S2] This is the second speaker's response.
```

### Example with Provided Sample

The repository includes `example_prompt.mp3` with transcript:
```
[S1] Open weights text to dialogue model.
[S2] You get full control over scripts and voices.
```

Use it directly:
```bash
python src/dia/podcast_generator_final.py \
    --url "https://example.com/article" \
    --voice-clone-audio example_prompt.mp3 \
    --output podcast_with_example_voices.wav
```

## Natural Speech Parameters

### Recommended Settings

For optimal natural speech:
- **Speed**: 0.85 (15% slower than original)
- **CFG Scale**: 3.5 for voice cloning, 2.8 for standard
- **Temperature**: 1.4 (controlled randomness)
- **Top-p**: 0.92 (balanced variety)

### Fine-tuning Speech Quality

Adjust parameters in the code for different effects:
```python
# For more expressive speech
cfg_scale=2.5

# For more consistent tone
temperature=1.2

# For slower pace
speed=0.8
```

## Output Files

The script generates:
1. **Audio File** (`.wav`): The final podcast audio
2. **Dialogue File** (`generated_dialogue.txt`): The script with speaker tags

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use `--cpu` flag
   - Reduce max_tokens in code

2. **Slow Generation**
   - Use GPU acceleration
   - Reduce dialogue length

3. **AWS Bedrock Errors**
   - Check AWS credentials
   - Verify Bedrock access in your region
   - Ensure Claude 3.5 Sonnet is available

4. **Voice Cloning Not Working**
   - Verify audio file path
   - Ensure transcript matches audio
   - Check audio quality and duration

### Performance Tips

- Use GPU for 5-10x faster generation
- Voice cloning adds ~20% to generation time
- First run downloads models (~6GB)

## 🤖 AWS Bedrock & Claude Integration

This project leverages **Claude 3.5 Sonnet** through **AWS Bedrock** for intelligent dialogue generation. The integration creates natural, engaging conversations from any web content.

### How It Works

1. **Content Analysis**: The system fetches and extracts key information from web articles
2. **AI Dialogue Generation**: Claude 3.5 Sonnet transforms the content into natural conversation
3. **Personality-Driven Speakers**: 
   - **S1**: Analytical and informative, provides context and explanations
   - **S2**: Curious and reactive, asks questions and shows enthusiasm
4. **Natural Speech Patterns**: Claude adds pauses, filler words, and conversational elements

### Bedrock Configuration

```python
# Initialize Bedrock client
self.bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)

# Use Claude 3.5 Sonnet v2
self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Request configuration
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 8000,
    "temperature": 0.8,
    "messages": [{
        "role": "user",
        "content": prompt
    }]
}
```

### Authentication Methods

#### 1. Local Development (AWS CLI)
```bash
# Configure AWS credentials
aws configure
# Set region (us-west-2 recommended)
# Enter access key and secret key
```

#### 2. EC2 with IAM Roles (Recommended for Production)
```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["bedrock:InvokeModel"],
        "Resource": "arn:aws:bedrock:*:*:model/anthropic.claude-3-5-sonnet*"
    }]
}
```

### Claude's Dialogue Generation Process

1. **Content Understanding**: Analyzes article structure, key points, and narrative flow
2. **Dialogue Planning**: Determines conversation flow and information distribution
3. **Natural Language**: Creates realistic back-and-forth with proper pacing
4. **Quality Control**: Ensures all key points are covered conversationally

### Example Prompt Structure

```python
prompt = f"""
Transform this content into a conversational podcast between two speakers.

Article Title: {title}
Article Content: {content[:8000]}

Guidelines:
- Create natural conversation, not a script
- Add pauses with "..." and commas for natural speech rhythm
- Include filler words ("well", "you know", "I mean") sparingly
- S1 should be analytical and informative
- S2 should be curious and reactive
- Cover all key points while maintaining engaging flow
- Target duration: {duration} minutes
"""
```

### Benefits of Using Claude via Bedrock

- **Contextual Understanding**: Deep comprehension of technical and complex topics
- **Natural Conversation**: Creates authentic dialogue that sounds human
- **Consistent Quality**: Reliable output across different content types
- **Scalability**: Handles articles of varying lengths and complexities
- **No Token Limits**: Bedrock handles long content seamlessly

## Architecture

The complete system architecture:
1. **Content Extraction**: BeautifulSoup for web scraping
2. **Dialogue Generation**: Claude 3.5 Sonnet via AWS Bedrock
3. **Speech Synthesis**: Dia-1.6B or Kokoro-82M models
4. **Voice Processing**: Voice cloning (Dia) or voice selection (Kokoro)
5. **Post-processing**: Speed adjustment and audio formatting

## License and Credits

- Dia-1.6B model by Nari Labs
- Claude 3.5 Sonnet by Anthropic
- Created with assistance from Claude

## Support

For issues or questions:
- Check the troubleshooting section
- Review example commands
- Ensure all dependencies are installed correctly