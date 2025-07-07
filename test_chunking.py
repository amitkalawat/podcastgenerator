#!/usr/bin/env python3
"""Test script to debug dialogue chunking issue"""

def split_dialogue_into_chunks(dialogue_text, chunk_size=15):
    """Split dialogue into manageable chunks for TTS generation"""
    lines = dialogue_text.strip().split('\n')
    chunks = []
    
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def analyze_dialogue_file():
    # Read the dialogue file
    with open('generated_dialogue.txt', 'r') as f:
        content = f.read()
    
    # Find where dialogue starts
    dialogue_start = content.find('[S1]')
    if dialogue_start == -1:
        print("No dialogue found!")
        return
    
    # Extract just the dialogue portion
    dialogue_text = content[dialogue_start:]
    
    # Remove any trailing content after the last dialogue line
    last_separator = dialogue_text.rfind('==================================================')
    if last_separator != -1:
        dialogue_text = dialogue_text[:last_separator].strip()
    
    print("Full dialogue text extracted:")
    print("="*50)
    print(dialogue_text[:200] + "...")
    print("="*50)
    
    # Split into lines
    lines = dialogue_text.strip().split('\n')
    print(f"\nTotal dialogue lines: {len(lines)}")
    
    # Show line 8 (index 7) which was reported missing
    print(f"\nLine 8 (index 7): {lines[7] if len(lines) > 7 else 'NOT FOUND'}")
    print(f"Line 16 (index 15): {lines[15] if len(lines) > 15 else 'NOT FOUND'}")
    
    # Split into chunks
    chunks = split_dialogue_into_chunks(dialogue_text, chunk_size=15)
    print(f"\nNumber of chunks: {len(chunks)}")
    
    # Analyze each chunk
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.strip().split('\n')
        print(f"\nChunk {i+1}: {len(chunk_lines)} lines")
        print(f"First line: {chunk_lines[0] if chunk_lines else 'EMPTY'}")
        print(f"Last line: {chunk_lines[-1] if chunk_lines else 'EMPTY'}")
        
        # Check if the missing lines are in this chunk
        for line in chunk_lines:
            if "What can it actually do though" in line:
                print(f"  >>> FOUND MISSING LINE IN CHUNK {i+1}: {line}")

if __name__ == "__main__":
    analyze_dialogue_file()