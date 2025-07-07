#!/usr/bin/env python3
"""Test what text is actually sent to TTS for each chunk"""

def split_dialogue_into_chunks(dialogue_text, chunk_size=15):
    """Split dialogue into manageable chunks for TTS generation"""
    lines = dialogue_text.strip().split('\n')
    chunks = []
    
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def simulate_tts_input():
    # Voice clone settings
    voice_clone_transcript = "[S1] Open weights text to dialogue model. [S2] You get full control over scripts and voices."
    
    # Read dialogue from file
    with open('generated_dialogue.txt', 'r') as f:
        content = f.read()
    
    # Extract dialogue (same as generate_dialogue_with_bedrock would return)
    marker = "Generated Dialogue:\n"
    start = content.find(marker)
    dialogue_section = content[start + len(marker):]
    end = dialogue_section.find("\n\n" + "="*50)
    dialogue_text = dialogue_section[:end]
    
    # Split into chunks
    chunks = split_dialogue_into_chunks(dialogue_text, chunk_size=15)
    
    print(f"Total chunks: {len(chunks)}")
    print("="*80)
    
    # Simulate what's sent to TTS for each chunk
    for i, chunk in enumerate(chunks):
        print(f"\nCHUNK {i+1}:")
        print("-"*40)
        
        if i == 0:
            # First chunk with voice cloning
            full_text = voice_clone_transcript.strip() + "\n" + chunk
            print("TEXT SENT TO TTS (with voice clone prepended):")
            print(full_text)
            
            # Count lines
            lines_sent = full_text.strip().split('\n')
            print(f"\nTotal lines sent to TTS: {len(lines_sent)}")
            
            # Check if our missing line is here
            for j, line in enumerate(lines_sent):
                if "What can it actually do though" in line:
                    print(f">>> MISSING LINE FOUND at position {j+1} in chunk {i+1}")
        else:
            # Subsequent chunks
            print("TEXT SENT TO TTS:")
            print(chunk)
            
            lines_sent = chunk.strip().split('\n')
            print(f"\nTotal lines sent to TTS: {len(lines_sent)}")
        
        print("="*80)

if __name__ == "__main__":
    simulate_tts_input()