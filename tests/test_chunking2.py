#!/usr/bin/env python3
"""Test script to debug dialogue chunking issue - version 2"""

def test_dialogue_extraction():
    # Read the dialogue file
    with open('generated_dialogue.txt', 'r') as f:
        full_content = f.read()
    
    print("FULL FILE CONTENT ANALYSIS:")
    print("="*50)
    
    # Find the "Generated Dialogue:" marker
    dialogue_marker = "Generated Dialogue:\n"
    marker_pos = full_content.find(dialogue_marker)
    
    if marker_pos != -1:
        # Extract everything after "Generated Dialogue:\n"
        after_marker = full_content[marker_pos + len(dialogue_marker):]
        
        # Find the ending separator
        end_separator = "\n\n" + "="*50
        end_pos = after_marker.find(end_separator)
        
        if end_pos != -1:
            dialogue_only = after_marker[:end_pos]
        else:
            dialogue_only = after_marker
        
        print("Extracted dialogue (first 500 chars):")
        print(dialogue_only[:500])
        print("\n" + "="*50)
        
        # Count lines
        dialogue_lines = dialogue_only.strip().split('\n')
        print(f"\nTotal dialogue lines (should be 33): {len(dialogue_lines)}")
        
        # Show specific lines
        print("\nChecking specific dialogue lines:")
        for i in [0, 7, 8, 15, 16]:
            if i < len(dialogue_lines):
                print(f"Line {i+1} (index {i}): {dialogue_lines[i][:80]}...")
        
        # Check for the missing content
        print("\nSearching for missing content:")
        for i, line in enumerate(dialogue_lines):
            if "What can it actually do though" in line:
                print(f"Found at line {i+1} (index {i}): {line}")
                
        return dialogue_only
    else:
        print("Could not find 'Generated Dialogue:' marker!")
        return None

def compare_with_generate_dialogue_method():
    """Simulate how the generate_dialogue_with_bedrock method returns dialogue"""
    
    # Read the dialogue file
    with open('generated_dialogue.txt', 'r') as f:
        content = f.read()
    
    # Find the "Generated Dialogue:" section
    marker = "Generated Dialogue:\n"
    start = content.find(marker)
    if start != -1:
        dialogue_section = content[start + len(marker):]
        
        # Find the ending separator
        end = dialogue_section.find("\n\n" + "="*50)
        if end != -1:
            dialogue_text = dialogue_section[:end]
        else:
            dialogue_text = dialogue_section
            
        # This should match what generate_dialogue_with_bedrock returns
        lines = dialogue_text.strip().split('\n')
        print(f"\nDialogue lines returned by generate_dialogue_with_bedrock: {len(lines)}")
        print("First line:", lines[0] if lines else "NONE")
        print("Line 8:", lines[7] if len(lines) > 7 else "NOT FOUND")
        
        return dialogue_text

if __name__ == "__main__":
    print("TEST 1: Extracting dialogue from file")
    dialogue1 = test_dialogue_extraction()
    
    print("\n\nTEST 2: Simulating generate_dialogue_with_bedrock return value")
    dialogue2 = compare_with_generate_dialogue_method()