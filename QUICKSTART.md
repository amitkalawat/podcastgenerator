# Quick Start Guide

## 1. Installation (5 minutes)

```bash
# Clone repository
git clone <repository-url>
cd podcastgenerator

# Install dependencies
pip install torch transformers scipy numpy beautifulsoup4 requests boto3 soundfile
pip install git+https://github.com/nari-labs/dia.git

# Configure AWS (if not already done)
aws configure
```

## 2. First Podcast (2 minutes)

Generate your first podcast:
```bash
python podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --output my_first_podcast.wav
```

## 3. Natural Speech Version

For better quality with natural pacing:
```bash
python podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --speed 0.85 \
    --output natural_podcast.wav
```

## 4. Voice Cloning

Clone voices from the included sample:
```bash
python podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --voice-clone-audio example_prompt.mp3 \
    --speed 0.85 \
    --output cloned_podcast.wav
```

## What to Expect

- **Generation Time**: 2-5 minutes depending on hardware
- **Output**: WAV audio file with 2 speakers discussing the article
- **Quality**: Natural conversational podcast with proper pacing

## Next Steps

- Try different articles by changing the URL
- Experiment with speed settings (0.8-1.0)
- Use your own audio samples for voice cloning
- Check generated_dialogue.txt to see the script