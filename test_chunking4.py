#!/usr/bin/env python3
"""Analyze what might be getting cut off in chunk 1"""

def analyze_chunk1_issue():
    # Voice clone transcript
    voice_clone_transcript = "[S1] Open weights text to dialogue model. [S2] You get full control over scripts and voices."
    
    # The dialogue lines that should be in chunk 1
    chunk1_lines = [
        "[S1] Hey, have you heard about the big news from Anthropic? They just announced Claude 4...",
        "[S2] No, what's that all about? I'm always curious about new AI developments.",
        "[S1] Well, they're actually launching two new models... Claude Opus 4 and Claude Sonnet 4.",
        "[S2] Oh interesting! What makes these different from previous versions?",
        "[S1] So, get this... Claude Opus 4 is apparently the world's best coding model right now. It's, you know, really impressive with complex tasks.",
        "[S2] Best coding model? That's a pretty big claim...",
        "[S1] Yeah, and they've got the numbers to back it up... It's scoring 72.5% on something called SWE-bench, which is, like, a major benchmark for coding.",
        "[S2] What can it actually do though, in practical terms?",  # Line 8 - MISSING
        "[S1] Well... it can work continuously for several hours, handling really complex problems. And, what's really cool is that it can use tools like web search while it's thinking.",
        "[S2] Hold on... it can search the web while working? How's that different from before?",
        "[S1] So, this is new... It can actually alternate between reasoning and using tools, which makes its responses much better. And get this... it can use multiple tools at the same time.",
        "[S2] That sounds pretty advanced. What about the other model, Sonnet 4?",
        "[S1] Right, so Sonnet 4 is like... the more practical version. It's still super capable, but it's designed more for everyday use.",
        "[S2] Who gets access to these models?",
        "[S1] Well... Opus 4 and Sonnet 4 are available on their Pro, Max, Team, and Enterprise plans. But, you know, Sonnet 4 is also available to free users."
    ]
    
    # Full text sent to TTS for chunk 1
    full_chunk1 = voice_clone_transcript + "\n" + "\n".join(chunk1_lines)
    
    print("CHUNK 1 ANALYSIS:")
    print("="*80)
    print(f"Voice clone transcript length: {len(voice_clone_transcript)} chars")
    print(f"Number of dialogue lines in chunk 1: {len(chunk1_lines)}")
    print(f"Total lines sent to TTS: {len(chunk1_lines) + 1}")
    
    # Calculate character count
    total_chars = len(full_chunk1)
    print(f"\nTotal characters in chunk 1: {total_chars}")
    
    # Estimate tokens (rough approximation: ~4 chars per token)
    estimated_tokens = total_chars / 4
    print(f"Estimated tokens (at ~4 chars/token): {estimated_tokens:.0f}")
    
    # Show where line 8 appears
    print("\nLine 8 position analysis:")
    lines_before_8 = chunk1_lines[:7]
    chars_before_8 = len(voice_clone_transcript + "\n" + "\n".join(lines_before_8))
    print(f"Characters before line 8: {chars_before_8}")
    print(f"Estimated tokens before line 8: {chars_before_8 / 4:.0f}")
    
    # Check if we're close to token limit
    print("\nToken limit analysis:")
    print(f"Max tokens setting: 2500")
    print(f"Estimated tokens in chunk 1: {estimated_tokens:.0f}")
    if estimated_tokens > 2500:
        print(">>> WARNING: Chunk 1 may exceed token limit!")
    
    # Check what happens if we reduce chunk size
    print("\n" + "="*80)
    print("ALTERNATIVE CHUNK SIZES:")
    
    for chunk_size in [10, 12, 13]:
        print(f"\nWith chunk_size={chunk_size}:")
        chunk1_alt = "\n".join(chunk1_lines[:chunk_size])
        full_text_alt = voice_clone_transcript + "\n" + chunk1_alt
        chars_alt = len(full_text_alt)
        tokens_alt = chars_alt / 4
        print(f"  Characters: {chars_alt}, Estimated tokens: {tokens_alt:.0f}")
        
        # Check if line 8 would be included
        if chunk_size >= 8:
            print(f"  Line 8 INCLUDED in chunk 1")
        else:
            print(f"  Line 8 would be in chunk 2")

if __name__ == "__main__":
    analyze_chunk1_issue()