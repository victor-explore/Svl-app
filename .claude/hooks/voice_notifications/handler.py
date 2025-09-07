#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pygame",
# ]
# ///
"""
Simplified Voice Notification Hook for Claude Code
=================================================

Plays completion notifications when Claude Code finishes tasks.
Only handles Stop and SubagentStop events with basic sound playback.

Author: Chong-U (chong-u@aioriented.dev)
Created: 2025
Purpose: Simple completion notification system for Claude Code

Usage: 
  python handler.py --voice=alfred
"""

import json
import sys
import argparse
import random
from pathlib import Path

def get_sound_name(hook_event_name, input_data=None):
    """Map hook event to sound file name with hardcoded mappings."""
    # Handle completion events with hardcoded mappings
    if hook_event_name == "Stop":
        completion_sounds = [
            "task_complete", "work_finished", "request_fulfilled", 
            "work_concluded", "assignment_finished"
        ]
        return random.choice(completion_sounds)
    elif hook_event_name == "SubagentStop":
        return "task_complete"
    else:
        # Default fallback
        return "task_complete"

def play_voice_sound(sound_name="task_complete"):
    """
    Play a voice sound using pygame library.
    
    Args:
        sound_name: Sound file name (task_complete, work_finished, etc.)
    """
    try:
        import pygame
        import os
        import time
        
        # Get script directory and build sound path
        script_dir = Path(__file__).parent
        
        # Try sound files directly in sounds directory
        mp3_path = script_dir / "sounds" / f"{sound_name}.mp3"
        wav_path = script_dir / "sounds" / f"{sound_name}.wav"
        
        if mp3_path.exists():
            sound_path = mp3_path
        elif wav_path.exists():
            sound_path = wav_path
        else:
            # No Alfred sounds found - use terminal bell
            print("\a", end="", flush=True)
            return
        
        # Set environment variable to suppress pygame messages
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        
        # Initialize pygame mixer
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
        pygame.mixer.init()
        
        # Load and play the sound
        sound = pygame.mixer.Sound(str(sound_path))
        sound.play()
        
        # Wait for sound to play completely
        time.sleep(0.1)  # Let it start
        
        # Wait for completion with timeout
        timeout = 3.0  # 3 seconds max wait
        start_time = time.time()
        
        while pygame.mixer.get_busy() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        pygame.mixer.quit()
        
    except ImportError:
        print("\a", end="", flush=True)  # Terminal bell fallback
        
    except Exception:
        print("\a", end="", flush=True)  # Terminal bell fallback

def main():
    """
    Main function - reads Claude's JSON hook data and plays appropriate sound.
    """
    parser = argparse.ArgumentParser(description='Play completion notifications for Claude Code')
    parser.add_argument('--voice', default='ignored', help='Voice parameter (ignored - kept for compatibility)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Read hook data from stdin (Claude provides this)
    try:
        input_data = json.load(sys.stdin)
        hook_event_name = input_data.get("hook_event_name", "Stop")
        
        # Only process completion events
        if hook_event_name in ["Stop", "SubagentStop"]:
            sound_name = get_sound_name(hook_event_name, input_data)
        else:
            # Ignore other events (PreToolUse, PostToolUse, Notification, etc.)
            sys.exit(0)
        
    except (json.JSONDecodeError, Exception):
        # Default to completion sound
        sound_name = "task_complete"
    
    # Play the sound
    play_voice_sound(sound_name)
    sys.exit(0)

if __name__ == "__main__":
    main()