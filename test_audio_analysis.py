#!/usr/bin/env python3
"""Analyze potential audio generation issues"""

def check_for_chunk_overlap_issues():
    """Check if there might be issues with how chunks are split"""
    
    # Read dialogue from file
    with open('generated_dialogue.txt', 'r') as f:
        content = f.read()
    
    # Extract dialogue
    marker = "Generated Dialogue:\n"
    start = content.find(marker)
    dialogue_section = content[start + len(marker):]
    end = dialogue_section.find("\n\n" + "="*50)
    dialogue_text = dialogue_section[:end]
    
    # Split into lines
    lines = dialogue_text.strip().split('\n')
    
    print("CHUNK BOUNDARY ANALYSIS:")
    print("="*80)
    print(f"Total dialogue lines: {len(lines)}")
    
    # Show chunk boundaries with chunk_size=15
    chunk_size = 15
    num_chunks = (len(lines) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(lines))
        
        print(f"\nCHUNK {chunk_idx + 1} (lines {start_idx + 1}-{end_idx}):")
        print("-" * 40)
        
        # Show first and last few lines of chunk
        chunk_lines = lines[start_idx:end_idx]
        
        if len(chunk_lines) <= 5:
            for i, line in enumerate(chunk_lines):
                print(f"  Line {start_idx + i + 1}: {line[:60]}...")
        else:
            # Show first 2
            for i in range(2):
                print(f"  Line {start_idx + i + 1}: {chunk_lines[i][:60]}...")
            print("  ...")
            # Show last 2
            for i in range(len(chunk_lines) - 2, len(chunk_lines)):
                print(f"  Line {start_idx + i + 1}: {chunk_lines[i][:60]}...")
        
        # Check if line 8 is in this chunk
        if start_idx < 8 <= end_idx:
            print(f"\n  >>> LINE 8 IS IN THIS CHUNK at position {8 - start_idx}")
            print(f"      Content: {lines[7]}")

def test_hypothesis():
    """Test hypothesis about what might be happening"""
    
    print("\n\nHYPOTHESIS TESTING:")
    print("="*80)
    
    print("\nPossible issues:")
    print("1. The TTS model might be cutting off content when it sees certain patterns")
    print("2. The voice clone transcript prepended to chunk 1 might affect processing")
    print("3. There might be a timing/synchronization issue with chunk generation")
    print("4. The audio chunk might be generated but not properly saved/concatenated")
    
    print("\nRecommended fixes to try:")
    print("1. Reduce chunk_size to 10-12 lines to ensure no truncation")
    print("2. Add logging to track exact content sent to TTS and received back")
    print("3. Save individual chunk audio files for debugging")
    print("4. Try generating without voice cloning to isolate the issue")
    print("5. Add small overlap between chunks (1-2 lines) to ensure continuity")

if __name__ == "__main__":
    check_for_chunk_overlap_issues()
    test_hypothesis()