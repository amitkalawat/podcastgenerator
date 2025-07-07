#!/usr/bin/env python3
"""
Main entry point for the Podcast Generator
Helps users choose between Dia-1.6B and Kokoro-82M implementations
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Podcast Generator - Choose your TTS model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Dia-1.6B with voice cloning
  python generate_podcast.py --model dia --voice-clone samples/example_prompt.mp3 --url https://example.com

  # Use Kokoro-82M with natural voices
  python generate_podcast.py --model kokoro --url https://example.com

  # Quick start with Kokoro (recommended for beginners)
  python generate_podcast.py --model kokoro --url https://example.com --quick

For more options, see the README.md or use --help with specific implementations.
        """
    )
    
    parser.add_argument(
        "--model",
        choices=["dia", "kokoro"],
        required=True,
        help="Choose TTS model: 'dia' for Dia-1.6B (voice cloning) or 'kokoro' for Kokoro-82M (no limits)"
    )
    
    parser.add_argument(
        "--implementation",
        help="Specific implementation to use (e.g., 'small_chunks' for Dia, 'ec2' for Kokoro)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use recommended settings for quick generation"
    )
    
    # Parse only known args to pass the rest to the actual implementation
    args, remaining = parser.parse_known_args()
    
    # Construct the command based on model choice
    if args.model == "dia":
        if args.implementation:
            script = f"src/dia/podcast_generator_{args.implementation}.py"
        else:
            # Default to small_chunks as it's most reliable
            script = "src/dia/podcast_generator_small_chunks.py"
            
        if args.quick:
            remaining.extend(["--speed", "0.85"])
            
    elif args.model == "kokoro":
        if args.implementation == "ec2":
            script = "src/kokoro/podcast_generator_kokoro_ec2.py"
        else:
            script = "src/kokoro/podcast_generator_kokoro.py"
            
        if args.quick:
            remaining.extend(["--s1-voice", "af_nova", "--s2-voice", "am_liam", "--duration", "3"])
    
    # Check if script exists
    if not os.path.exists(script):
        print(f"Error: Script '{script}' not found!")
        print("\nAvailable implementations:")
        if args.model == "dia":
            print("  - final")
            print("  - chunked")
            print("  - chunked_fixed")
            print("  - small_chunks (recommended)")
            print("  - fixed_trimming")
            print("  - no_overlap")
            print("  - consistent_voices")
        else:
            print("  - kokoro (default)")
            print("  - ec2")
        sys.exit(1)
    
    # Execute the chosen implementation
    import subprocess
    cmd = [sys.executable, script] + remaining
    
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()