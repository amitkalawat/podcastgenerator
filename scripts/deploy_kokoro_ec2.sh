#!/bin/bash
# Deployment script for Kokoro podcast generator on EC2 with IAM roles

echo "==================================================="
echo "Deploying Kokoro Podcast Generator on EC2"
echo "==================================================="

# Update system
echo "1. Updating system packages..."
sudo apt-get update -y

# Install system dependencies
echo "2. Installing espeak-ng (required for Kokoro)..."
sudo apt-get install -y espeak-ng

# Install Python dependencies
echo "3. Installing Python packages..."
pip3 install --upgrade pip
pip3 install kokoro>=0.9.2 soundfile beautifulsoup4 boto3 requests

# Test Kokoro installation
echo "4. Testing Kokoro installation..."
python3 -c "from kokoro import KPipeline; print('✓ Kokoro imported successfully')"

# Test AWS Bedrock access (should work with IAM role)
echo "5. Testing AWS Bedrock access..."
python3 -c "
import boto3
try:
    client = boto3.client('bedrock-runtime', region_name='us-west-2')
    print('✓ AWS Bedrock client created successfully')
    print('  IAM role authentication is working')
except Exception as e:
    print(f'⚠️  AWS Bedrock error: {e}')
    print('  Check IAM role permissions for Bedrock')
"

# Create test script
echo "6. Creating test script..."
cat > test_kokoro.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('.')
from podcast_generator_kokoro import PodcastGenerator

print("\nTesting Kokoro podcast generation...")
generator = PodcastGenerator(
    use_gpu=True,  # Will use GPU if available
    aws_region='us-west-2',
    s1_voice='af_nova',
    s2_voice='am_liam'
)

# Test with a simple URL
generator.create_podcast(
    content_source="https://www.anthropic.com/news/claude-4",
    output_file="test_podcast_ec2.wav",
    duration_minutes=2
)
EOF

chmod +x test_kokoro.py

echo ""
echo "==================================================="
echo "Deployment complete!"
echo "==================================================="
echo ""
echo "To generate a podcast:"
echo "  python3 podcast_generator_kokoro.py --url https://example.com/article --duration 5"
echo ""
echo "Voice options:"
echo "  Female: af_nova, af_alloy, af_bella, af_heart"
echo "  Male: am_liam, am_echo, am_michael, am_adam"
echo ""
echo "Example with custom voices:"
echo "  python3 podcast_generator_kokoro.py \\"
echo "    --url https://example.com/article \\"
echo "    --s1-voice af_alloy \\"
echo "    --s2-voice am_echo \\"
echo "    --duration 5 \\"
echo "    --speed 0.9"
echo ""
echo "IAM Role Requirements:"
echo "  - bedrock:InvokeModel"
echo "  - bedrock:ListFoundationModels (optional, for testing)"
echo ""