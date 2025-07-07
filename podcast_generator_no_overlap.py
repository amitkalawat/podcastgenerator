#!/usr/bin/env python3
"""
Podcast Generator with Dia-1.6B TTS and Claude 3.5 Sonnet via AWS Bedrock
Version 5.7 - No Overlap Strategy with Better Voice Consistency
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
import torch
import boto3
import json
import random
import warnings
from pathlib import Path
import numpy as np
import soundfile as sf

# Import Dia model - will need to be installed separately
try:
    from dia.model import Dia
    DIA_AVAILABLE = True
except ImportError:
    DIA_AVAILABLE = False
    print("Warning: Dia module not found. Voice cloning features will be disabled.")
    print("Install with: pip install git+https://github.com/nari-labs/dia.git")

warnings.filterwarnings('ignore')

# TTS Model
TTS_MODEL = "nari-labs/Dia-1.6B-0626"

class PodcastGenerator:
    def __init__(self, use_gpu=True, aws_region='us-west-2', seed=None, voice_clone_audio=None, voice_clone_transcript=None, speech_speed=0.85):
        """Initialize the podcast generator with optional voice cloning and speed control"""
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Set seed if provided
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(f"✓ Random seed set to: {seed}")
        
        # Speech speed (0.8-1.0, where lower is slower)
        self.speech_speed = speech_speed
        print(f"✓ Speech speed set to: {speech_speed}x")
        
        # Voice cloning configuration
        self.voice_clone_audio = voice_clone_audio
        self.voice_clone_transcript = voice_clone_transcript
        
        if self.voice_clone_audio and DIA_AVAILABLE:
            print(f"✓ Voice cloning enabled with: {self.voice_clone_audio}")
            if self.voice_clone_transcript:
                print(f"✓ Voice clone transcript loaded")
        
        # Initialize Bedrock client
        print("Initializing AWS Bedrock client...")
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )
        self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        print(f"Using Claude 3.5 Sonnet model: {self.model_id}")
        
        # Load Dia model if available
        if DIA_AVAILABLE:
            print("Loading Dia-1.6B TTS model...")
            try:
                # Use float16 for GPU, float32 for CPU
                compute_dtype = "float16" if self.device == "cuda" else "float32"
                self.tts_model = Dia.from_pretrained(TTS_MODEL, compute_dtype=compute_dtype)
                print("✓ Dia-1.6B model loaded successfully")
            except Exception as e:
                print(f"Error loading Dia-1.6B: {e}")
                raise
        else:
            print("⚠️  Dia model not available - using fallback TTS")
            self.tts_model = None
    
    def adjust_audio_speed(self, audio_data, sample_rate, speed_factor):
        """Adjust audio playback speed using resampling"""
        # Calculate new length
        new_length = int(len(audio_data) / speed_factor)
        
        # Use linear interpolation to resample
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        resampled_audio = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        return resampled_audio.astype(audio_data.dtype)
    
    def extract_content_from_url(self, url):
        """Extract text content from a URL"""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract title
            title = soup.find('title')
            title = title.string if title else "Web Content"
            
            # Return more content for longer podcasts
            return title, text[:20000]  # Increased from 10000
        except Exception as e:
            print(f"Error extracting content: {e}")
            return "Content", "Failed to extract content from URL."
    
    def generate_dialogue_with_bedrock(self, content, title, duration_minutes=5):
        """Generate a dialogue between two hosts using Claude 3.5 Sonnet via Bedrock"""
        # Calculate approximate exchanges needed (assuming ~3-4 seconds per exchange)
        target_exchanges = int(duration_minutes * 60 / 3.5)
        
        prompt = f"""You are a professional podcast content generator. Your job is to analyze content and create natural, engaging podcast dialogues between two speakers.

Source Material:
Article Title: {title}
Article Content: {content[:8000]}...

Your task: Transform this content into a conversational podcast between two speakers using [S1] and [S2] tags.

IMPORTANT: Add natural pauses and pacing cues for slower, more natural speech:
- Use "..." for brief pauses
- Use commas liberally for natural breathing points
- Break longer sentences into shorter chunks
- Add filler words occasionally like "well", "you know", "I mean"

Target Duration: Generate approximately {target_exchanges} conversational exchanges (about {duration_minutes} minutes of audio)

Example format:
[S1] Have you heard about this new development in AI? It's... quite fascinating.
[S2] No, tell me more about it. I'm curious.
[S1] Well, it's quite interesting... They've created a model that can, you know, understand context better.
[S2] That sounds incredible! How does it work, exactly?

Dialogue Guidelines:
- Generate {target_exchanges-5} to {target_exchanges+5} short conversational exchanges
- Keep responses concise and natural (1-2 sentences typical)
- Add natural pauses with "..." and commas
- Include occasional filler words for naturalness
- DO NOT include any emotional reactions in parentheses
- Make it sound like a real conversation, not a script
- Use natural flow and transitions
- Cover multiple aspects of the topic in depth

Speaker Personalities:
- [S1]: More analytical, provides facts and context, speaks thoughtfully
- [S2]: More reactive, asks questions, shows interest through words

Content Focus:
- Cover the main topic thoroughly
- Include ALL key points and details
- Discuss implications and future outlook
- Make complex topics accessible
- Add personal insights and reflections

Remember: This is a conversation between two people discussing interesting content. Keep it natural, well-paced, and engaging."""

        try:
            # Prepare the request for Bedrock
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8000,  # Increased from 2500
                "temperature": 0.8,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            # Make the request
            print("Calling Claude 3.5 Sonnet via Bedrock...")
            response = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response.get('body').read())
            dialogue_text = response_body['content'][0]['text']
            
            # Extract dialogue lines
            dialogue_lines = []
            for line in dialogue_text.split('\n'):
                line = line.strip()
                if line.startswith('[S1]') or line.startswith('[S2]'):
                    dialogue_lines.append(line)
            
            # Join all dialogue lines
            full_dialogue = '\n'.join(dialogue_lines)
            
            # Save dialogue to file
            dialogue_filename = "generated_dialogue.txt"
            with open(dialogue_filename, 'w', encoding='utf-8') as f:
                f.write(f"Generated Podcast Dialogue\n")
                f.write(f"Article: {title}\n")
                f.write(f"Speech Speed: {self.speech_speed}x\n")
                f.write(f"Target Duration: {duration_minutes} minutes\n")
                if self.voice_clone_audio:
                    f.write(f"Voice Clone Audio: {self.voice_clone_audio}\n")
                f.write("="*50 + "\n\n")
                
                if self.voice_clone_transcript:
                    f.write("Voice Clone Transcript:\n")
                    f.write(self.voice_clone_transcript + "\n")
                    f.write("-"*50 + "\n\n")
                
                f.write("Generated Dialogue:\n")
                f.write(full_dialogue)
                f.write("\n\n" + "="*50 + "\n")
                f.write(f"Total lines: {len(dialogue_lines)}\n")
            
            print(f"✓ Dialogue saved to: {dialogue_filename}")
            print(f"✓ Generated {len(dialogue_lines)} dialogue lines")
            
            return full_dialogue
            
        except Exception as e:
            print(f"Error with Bedrock: {e}")
            return self.generate_fallback_dialogue()
    
    def generate_fallback_dialogue(self):
        """Generate a fallback dialogue if Bedrock fails"""
        return """[S1] Welcome to our tech podcast! Today... we're discussing an exciting development.
[S2] What's the topic for today?
[S1] Well, we're looking at a major AI advancement that could, you know, change everything.
[S2] Tell me more about it!
[S1] It involves new technology that's... pushing the boundaries of what's possible.
[S2] That sounds fascinating. What are the key benefits?
[S1] The implications are, well... huge for various industries.
[S2] How soon will we see this in action?
[S1] Development is moving quickly, so... we might see results sooner than expected.
[S2] This could be a game-changer!
[S1] Absolutely. It's an exciting time for technology.
[S2] Thanks for breaking this down for our listeners!"""
    
    def create_smart_chunks(self, dialogue_text):
        """Create chunks at natural conversation breaks"""
        lines = dialogue_text.strip().split('\n')
        chunks = []
        current_chunk = []
        current_chunk_lines = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_chunk_lines += 1
            
            # Check if we should end this chunk
            # End chunk if: 1) we have 4-6 lines, 2) next line is different speaker, 3) natural break point
            should_end_chunk = False
            
            # If we have at least 4 lines
            if current_chunk_lines >= 4:
                # Check if next line exists and is different speaker
                if i + 1 < len(lines):
                    current_speaker = '[S1]' if '[S1]' in line else '[S2]'
                    next_speaker = '[S1]' if '[S1]' in lines[i + 1] else '[S2]'
                    
                    # End chunk if speaker changes or we have 6 lines
                    if current_speaker != next_speaker or current_chunk_lines >= 6:
                        should_end_chunk = True
                else:
                    # Last line
                    should_end_chunk = True
            
            # If this is the last line
            if i == len(lines) - 1:
                should_end_chunk = True
            
            if should_end_chunk and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_chunk_lines = 0
        
        # Log chunk information
        print(f"\nDialogue split into {len(chunks)} smart chunks:")
        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.count('\n') + 1
            speakers = []
            if '[S1]' in chunk:
                speakers.append('S1')
            if '[S2]' in chunk:
                speakers.append('S2')
            print(f"  Chunk {i+1}: {chunk_lines} lines, speakers: {', '.join(speakers)}")
        
        return chunks
    
    def generate_voice_establishment_audio(self):
        """Generate a short audio to establish voice characteristics"""
        if not self.voice_clone_audio or not self.voice_clone_transcript:
            return None
        
        try:
            print("\nGenerating voice establishment audio...")
            # Use the voice clone transcript plus a neutral continuation
            establishment_text = self.voice_clone_transcript.strip() + "\n" + """[S1] Let's begin our discussion today.
[S2] I'm looking forward to it."""
            
            output = self.tts_model.generate(
                establishment_text,
                audio_prompt=self.voice_clone_audio,
                use_torch_compile=False,
                verbose=False,
                cfg_scale=3.5,
                temperature=1.2,  # Slightly lower for establishment
                top_p=0.90,
                cfg_filter_top_k=40,
                max_tokens=1000
            )
            
            # Save to temp file
            temp_file = "voice_establishment.wav"
            self.tts_model.save_audio(temp_file, output)
            
            # Load and return just for voice "priming" - we won't use this audio
            audio_data, sample_rate = sf.read(temp_file)
            os.remove(temp_file)
            
            print("✓ Voice characteristics established")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to establish voice characteristics: {e}")
            return False
    
    def generate_audio_chunk(self, dialogue_chunk, chunk_index, total_chunks):
        """Generate audio for a single chunk without overlap"""
        if not DIA_AVAILABLE or not self.tts_model:
            print("⚠️  Dia model not available, skipping audio generation")
            return None
        
        try:
            print(f"\nGenerating chunk {chunk_index + 1}/{total_chunks}...")
            
            # For the first chunk, prepend voice clone transcript to establish voices
            if chunk_index == 0 and self.voice_clone_audio and self.voice_clone_transcript:
                # First chunk gets voice establishment
                full_text = self.voice_clone_transcript.strip() + "\n" + dialogue_chunk
                print("  Including voice clone transcript for voice establishment")
                needs_trimming = True
                trim_text = self.voice_clone_transcript.strip()
            else:
                # Other chunks use dialogue directly
                full_text = dialogue_chunk
                needs_trimming = False
                trim_text = ""
            
            # Log the exact text being processed
            print(f"  Text length: {len(full_text)} characters")
            print(f"  Chunk preview: {dialogue_chunk[:100]}...")
            
            # Generate with consistent parameters
            if self.voice_clone_audio:
                output = self.tts_model.generate(
                    full_text,
                    audio_prompt=self.voice_clone_audio,
                    use_torch_compile=False,
                    verbose=True,
                    cfg_scale=3.5,
                    temperature=1.2,    # Balanced for consistency
                    top_p=0.92,
                    cfg_filter_top_k=40,
                    max_tokens=3000     # Larger for smart chunks
                )
            else:
                output = self.tts_model.generate(
                    full_text,
                    use_torch_compile=False,
                    verbose=True,
                    cfg_scale=2.8,
                    temperature=1.2,
                    top_p=0.92,
                    cfg_filter_top_k=40,
                    max_tokens=3000
                )
            
            # Save chunk to temporary file
            temp_file = f"temp_chunk_{chunk_index}.wav"
            self.tts_model.save_audio(temp_file, output)
            
            # Load audio
            audio_data, sample_rate = sf.read(temp_file)
            
            # For first chunk only, trim the voice clone portion
            if needs_trimming and chunk_index == 0:
                # Estimate trim based on character ratio (rough approximation)
                trim_ratio = len(trim_text) / len(full_text)
                trim_samples = int(len(audio_data) * trim_ratio * 0.85)  # 85% of estimate
                audio_data = audio_data[trim_samples:]
                print(f"  Trimmed {trim_samples/sample_rate:.1f}s of voice establishment audio")
            
            # Apply speed adjustment
            if self.speech_speed != 1.0:
                audio_data = self.adjust_audio_speed(audio_data, sample_rate, self.speech_speed)
            
            # Log chunk duration
            duration = len(audio_data) / sample_rate
            print(f"  Chunk duration: {duration:.1f} seconds")
            
            # Save debug files
            debug_audio = f"debug_chunk_{chunk_index:03d}_audio.wav"
            debug_text = f"debug_chunk_{chunk_index:03d}_text.txt"
            
            sf.write(debug_audio, audio_data, sample_rate)
            with open(debug_text, 'w', encoding='utf-8') as f:
                f.write(f"Chunk {chunk_index + 1}/{total_chunks}\n")
                f.write("="*50 + "\n")
                f.write("Dialogue chunk:\n")
                f.write(dialogue_chunk)
                if needs_trimming:
                    f.write(f"\n\nNote: Voice clone transcript was prepended but trimmed from audio")
            
            print(f"  Debug files saved: {debug_audio}, {debug_text}")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Error generating audio chunk {chunk_index}: {e}")
            with open(f"debug_chunk_{chunk_index:03d}_error.txt", 'w') as f:
                f.write(f"Error: {str(e)}\n")
                f.write(f"Text that failed:\n{dialogue_chunk}")
            return None, None
    
    def concatenate_audio_chunks(self, audio_chunks, sample_rate, output_file):
        """Concatenate multiple audio chunks into a single file"""
        if not audio_chunks:
            return None
        
        # Add small silence between chunks (0.3 seconds)
        silence_duration = 0.3
        silence_samples = int(silence_duration * sample_rate)
        silence = np.zeros(silence_samples)
        
        # Concatenate with silence between chunks
        combined_chunks = []
        for i, chunk in enumerate(audio_chunks):
            combined_chunks.append(chunk)
            if i < len(audio_chunks) - 1:  # Don't add silence after last chunk
                combined_chunks.append(silence)
        
        combined_audio = np.concatenate(combined_chunks)
        
        # Save the final audio
        sf.write(output_file, combined_audio, sample_rate)
        
        duration = len(combined_audio) / sample_rate
        print(f"\n✓ Combined audio saved to: {output_file}")
        print(f"  Total duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
        print(f"  Total chunks combined: {len(audio_chunks)}")
        
        return output_file
    
    def generate_audio_with_dia(self, dialogue_text, output_file):
        """Generate audio using smart chunking without overlap"""
        if not DIA_AVAILABLE or not self.tts_model:
            print("⚠️  Dia model not available, skipping audio generation")
            return None
        
        try:
            # Optionally establish voices first (priming run)
            if self.voice_clone_audio:
                self.generate_voice_establishment_audio()
            
            # Create smart chunks at conversation breaks
            chunks = self.create_smart_chunks(dialogue_text)
            
            audio_chunks = []
            sample_rate = None
            successful_chunks = 0
            failed_chunks = 0
            
            # Generate audio for each chunk
            for i, chunk in enumerate(chunks):
                audio_data, sr = self.generate_audio_chunk(chunk, i, len(chunks))
                
                if audio_data is not None:
                    audio_chunks.append(audio_data)
                    successful_chunks += 1
                    if sample_rate is None:
                        sample_rate = sr
                else:
                    print(f"  ⚠️  Failed to generate chunk {i + 1}")
                    failed_chunks += 1
            
            print(f"\nGeneration summary:")
            print(f"  Successful chunks: {successful_chunks}/{len(chunks)}")
            print(f"  Failed chunks: {failed_chunks}/{len(chunks)}")
            
            # Concatenate all chunks
            if audio_chunks:
                return self.concatenate_audio_chunks(audio_chunks, sample_rate, output_file)
            else:
                print("Error: No audio chunks were generated successfully")
                return None
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    def create_podcast(self, content_source, output_file="podcast.wav", duration_minutes=5):
        """Create a complete podcast from content"""
        print("\n" + "="*50)
        print("Creating podcast (no overlap strategy)...")
        print("="*50 + "\n")
        
        # Extract content from URL
        print(f"1. Extracting content from: {content_source}")
        title, content = self.extract_content_from_url(content_source)
        print(f"   ✓ Article title: {title}")
        
        # Generate dialogue with Claude
        print(f"\n2. Generating dialogue with Claude 3.5 Sonnet...")
        print(f"   Target duration: {duration_minutes} minutes")
        dialogue = self.generate_dialogue_with_bedrock(content, title, duration_minutes)
        
        # Generate audio with Dia
        print("\n3. Generating audio with Dia-1.6B (smart chunks, no overlap)...")
        if self.voice_clone_audio:
            print(f"   Using voice clone from: {self.voice_clone_audio}")
        print(f"   Speech speed: {self.speech_speed}x")
        print(f"   Strategy: Smart chunking at speaker changes")
        print(f"   No overlap between chunks")
        
        audio_file = self.generate_audio_with_dia(dialogue, output_file)
        
        if audio_file:
            print(f"\n✓ Podcast successfully created: {output_file}")
        else:
            print("\n⚠️  Audio generation failed or not available")
        
        return audio_file

def main():
    parser = argparse.ArgumentParser(
        description="Generate conversational podcasts without chunk overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard generation
  python %(prog)s --url https://example.com/article --duration 5
  
  # With voice cloning and custom settings
  python %(prog)s --url https://example.com/article \\
    --voice-clone-audio example_prompt.mp3 \\
    --speed 0.9 \\
    --duration 10
        """
    )
    
    parser.add_argument("--url", 
                       default="https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster",
                       help="URL to generate podcast from")
    parser.add_argument("-o", "--output", 
                       default="podcast_output.wav",
                       help="Output audio file")
    parser.add_argument("--cpu", 
                       action="store_true",
                       help="Force CPU usage")
    parser.add_argument("--region", 
                       default="us-west-2",
                       help="AWS region for Bedrock")
    parser.add_argument("--seed", 
                       type=int,
                       help="Random seed for reproducible generation")
    parser.add_argument("--speed", 
                       type=float,
                       default=0.85,
                       help="Speech speed (0.8-1.0, default: 0.85 for natural pace)")
    parser.add_argument("--duration", 
                       type=int,
                       default=5,
                       help="Target podcast duration in minutes (default: 5)")
    
    # Voice cloning arguments
    voice_group = parser.add_argument_group('voice cloning options')
    voice_group.add_argument("--voice-clone-audio",
                           help="Path to audio file for voice cloning")
    voice_group.add_argument("--voice-clone-transcript",
                           help="Transcript of the voice clone audio")
    
    args = parser.parse_args()
    
    # Validate speed parameter
    if args.speed < 0.8 or args.speed > 1.0:
        print("Warning: Speed should be between 0.8 and 1.0. Using default 0.85")
        args.speed = 0.85
    
    # Handle default transcript for example_prompt.mp3
    voice_clone_transcript = args.voice_clone_transcript
    if args.voice_clone_audio and not voice_clone_transcript:
        if "example_prompt.mp3" in args.voice_clone_audio:
            voice_clone_transcript = "[S1] Open weights text to dialogue model. [S2] You get full control over scripts and voices."
            print("Using default transcript for example_prompt.mp3")
    
    # Create generator
    generator = PodcastGenerator(
        use_gpu=not args.cpu,
        aws_region=args.region,
        seed=args.seed,
        voice_clone_audio=args.voice_clone_audio,
        voice_clone_transcript=voice_clone_transcript,
        speech_speed=args.speed
    )
    
    # Generate podcast
    generator.create_podcast(args.url, args.output, args.duration)

if __name__ == "__main__":
    main()