#!/usr/bin/env python3
"""
Analyze the trimming issue in the first chunk of generated audio
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def find_silence_boundaries(audio_data, sample_rate, silence_threshold=0.01, min_silence_duration=0.1):
    """Find silence boundaries in audio data"""
    # Calculate RMS in windows
    window_size = int(sample_rate * 0.01)  # 10ms windows
    hop_size = window_size // 2
    
    rms_values = []
    for i in range(0, len(audio_data) - window_size, hop_size):
        window = audio_data[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    rms_values = np.array(rms_values)
    
    # Find silence regions
    silence_mask = rms_values < silence_threshold
    
    # Find transitions
    transitions = np.diff(silence_mask.astype(int))
    silence_starts = np.where(transitions == 1)[0]
    silence_ends = np.where(transitions == -1)[0]
    
    # Convert to samples
    silence_regions = []
    for start, end in zip(silence_starts, silence_ends):
        start_sample = start * hop_size
        end_sample = end * hop_size
        duration = (end_sample - start_sample) / sample_rate
        if duration >= min_silence_duration:
            silence_regions.append((start_sample, end_sample, duration))
    
    return silence_regions, rms_values

def analyze_audio_file(audio_file):
    """Analyze an audio file to find potential trim points"""
    print(f"\nAnalyzing: {audio_file}")
    
    # Load audio
    audio_data, sample_rate = sf.read(audio_file)
    duration = len(audio_data) / sample_rate
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Find silence regions
    silence_regions, rms_values = find_silence_boundaries(audio_data, sample_rate)
    
    print(f"\nFound {len(silence_regions)} significant silence regions:")
    for i, (start, end, dur) in enumerate(silence_regions):
        start_time = start / sample_rate
        end_time = end / sample_rate
        print(f"  Region {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {dur:.2f}s)")
    
    # Look for the first major silence after initial speech
    # This might indicate the boundary between voice clone and dialogue
    if len(silence_regions) > 0:
        # Find the first silence that's after at least 2 seconds of audio
        for start, end, dur in silence_regions:
            if start / sample_rate > 2.0:  # After first 2 seconds
                print(f"\nSuggested trim point: {end / sample_rate:.2f} seconds")
                print(f"This would preserve the first {start / sample_rate:.2f} seconds before the silence")
                break
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    time = np.arange(len(audio_data)) / sample_rate
    plt.plot(time, audio_data)
    plt.ylabel('Amplitude')
    plt.title(f'Waveform of {os.path.basename(audio_file)}')
    
    # Mark silence regions
    for start, end, _ in silence_regions:
        plt.axvspan(start/sample_rate, end/sample_rate, alpha=0.3, color='red')
    
    # Plot RMS
    plt.subplot(2, 1, 2)
    rms_time = np.arange(len(rms_values)) * 0.005  # 5ms hop
    plt.plot(rms_time, rms_values)
    plt.axhline(y=0.01, color='r', linestyle='--', label='Silence threshold')
    plt.ylabel('RMS')
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(audio_file)[0]}_analysis.png')
    print(f"\nVisualization saved to: {os.path.splitext(audio_file)[0]}_analysis.png")
    
    return silence_regions

def main():
    # Look for debug chunk files
    chunk_files = []
    for i in range(10):  # Check first 10 chunks
        filename = f"debug_chunk_{i:03d}_audio.wav"
        if os.path.exists(filename):
            chunk_files.append(filename)
    
    if not chunk_files:
        print("No debug chunk files found.")
        print("Please run the podcast generator first to create debug files.")
        return
    
    print(f"Found {len(chunk_files)} debug chunk files")
    
    # Analyze each chunk
    for chunk_file in chunk_files:
        analyze_audio_file(chunk_file)
    
    # Also analyze the final podcast if it exists
    podcast_files = ["podcast_output.wav", "aws_rainier_natural_speech_final.wav", 
                     "claude4_no_overlap_podcast.wav"]
    
    for podcast_file in podcast_files:
        if os.path.exists(podcast_file):
            print("\n" + "="*50)
            analyze_audio_file(podcast_file)

if __name__ == "__main__":
    main()