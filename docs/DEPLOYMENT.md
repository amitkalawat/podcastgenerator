# Deployment Guide

## EC2 Deployment

### 1. Instance Requirements

- **Type**: g4dn.xlarge or better (GPU required for speed)
- **AMI**: Deep Learning AMI with PyTorch
- **Storage**: 50GB+ (for models and audio files)
- **Region**: us-west-2 (or your Bedrock region)

### 2. Initial Setup

```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install additional dependencies
pip install beautifulsoup4 requests boto3 soundfile

# Install Dia
pip install git+https://github.com/nari-labs/dia.git
```

### 3. Deploy Code

```bash
# Copy files to EC2
scp -i your-key.pem podcast_generator_final.py ubuntu@your-ec2-ip:~/
scp -i your-key.pem example_prompt.mp3 ubuntu@your-ec2-ip:~/
scp -i your-key.pem requirements.txt ubuntu@your-ec2-ip:~/
```

### 4. Configure AWS Credentials

On EC2 instance:
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region (us-west-2)
# Enter default output format (json)
```

### 5. Test Deployment

```bash
# Basic test
python3 podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --output test_podcast.wav

# Voice cloning test
python3 podcast_generator_final.py \
    --url "https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster" \
    --voice-clone-audio example_prompt.mp3 \
    --speed 0.85 \
    --output test_cloned.wav
```

### 6. Download Results

From local machine:
```bash
scp -i your-key.pem ubuntu@your-ec2-ip:~/test_podcast.wav ./
```

## Production Deployment

### 1. Create Service Script

```bash
#!/bin/bash
# podcast_service.sh

URL=$1
OUTPUT=$2
VOICE_CLONE=${3:-""}

if [ -z "$VOICE_CLONE" ]; then
    python3 podcast_generator_final.py \
        --url "$URL" \
        --speed 0.85 \
        --output "$OUTPUT"
else
    python3 podcast_generator_final.py \
        --url "$URL" \
        --voice-clone-audio "$VOICE_CLONE" \
        --speed 0.85 \
        --output "$OUTPUT"
fi
```

### 2. API Wrapper (Optional)

Create a simple Flask API:
```python
# api.py
from flask import Flask, request, jsonify, send_file
import subprocess
import uuid
import os

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_podcast():
    data = request.json
    url = data.get('url')
    voice_clone = data.get('voice_clone_audio')
    
    output_file = f"podcast_{uuid.uuid4()}.wav"
    
    cmd = [
        'python3', 'podcast_generator_final.py',
        '--url', url,
        '--speed', '0.85',
        '--output', output_file
    ]
    
    if voice_clone:
        cmd.extend(['--voice-clone-audio', voice_clone])
    
    subprocess.run(cmd)
    
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 3. Systemd Service

```ini
# /etc/systemd/system/podcast-generator.service
[Unit]
Description=Podcast Generator Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/usr/bin/python3 /home/ubuntu/api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable podcast-generator
sudo systemctl start podcast-generator
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/nari-labs/dia.git

# Copy application files
COPY podcast_generator_final.py .
COPY example_prompt.mp3 .

# Set environment variables
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "podcast_generator_final.py"]
```

### Build and Run

```bash
# Build image
docker build -t podcast-generator .

# Run container
docker run --gpus all -v $(pwd)/output:/app/output podcast-generator \
    --url "https://example.com/article" \
    --output /app/output/podcast.wav
```

## Performance Optimization

### 1. GPU Optimization

- Use torch.compile for faster inference (if supported)
- Batch multiple requests when possible
- Use half-precision (float16) on compatible GPUs

### 2. Caching

- Cache extracted web content
- Store generated dialogues for similar content
- Keep model loaded in memory for API deployments

### 3. Scaling

- Use multiple GPU instances for parallel processing
- Implement queue system for large-scale generation
- Consider using AWS Batch for bulk processing

## Monitoring

### Basic Logging

Add to script:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcast_generator.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics to Track

- Generation time per podcast
- Success/failure rates
- Average file sizes
- GPU utilization
- Memory usage

## Security Considerations

1. **Input Validation**: Validate URLs before processing
2. **Rate Limiting**: Implement for API endpoints
3. **File Management**: Clean up temporary files
4. **AWS Credentials**: Use IAM roles instead of keys
5. **Network Security**: Restrict EC2 security groups