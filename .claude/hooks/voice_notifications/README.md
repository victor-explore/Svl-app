# Simple Voice Completion Notifications

Basic voice notification system for Claude Code that plays sounds when tasks are completed.

## Features

- **Completion notifications** - Audio feedback when Claude finishes tasks
- **Multiple completion sounds** - 5 completion sound variations for variety
- **Simple configuration** - Minimal setup required
- **Simple audio handling** - Uses completion sounds or terminal bell if unavailable

## Quick Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv add pygame

# Or using pip
pip install pygame
```

### 2. Configuration

Add to your `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "command": "uv run .claude/hooks/voice_notifications/handler.py"
      }]
    }],
    "SubagentStop": [{
      "hooks": [{
        "command": "uv run .claude/hooks/voice_notifications/handler.py"
      }]
    }]
  }
}
```

## Testing

Test completion notification:

```bash
echo '{"hook_event_name": "Stop"}' | uv run .claude/hooks/voice_notifications/handler.py
```

## Sound Files

The system uses these completion sounds:
- `task_complete.mp3` - Default completion sound
- `work_finished.mp3` - Alternative completion sound
- `work_concluded.mp3` - Alternative completion sound
- `assignment_finished.mp3` - Alternative completion sound  
- `request_fulfilled.mp3` - Alternative completion sound

If completion sounds are unavailable, the system falls back to a terminal bell.

## Events Supported

- **Stop** - Task completion (plays random completion sound)
- **SubagentStop** - Subtask completion (plays task_complete.mp3)

All other Claude Code events (PreToolUse, PostToolUse, Notification, etc.) are ignored.

## File Structure

Ultra-minimal structure with only essential files:

```
.claude/hooks/
└── voice_notifications/
    ├── handler.py (self-contained script with hardcoded mappings)
    ├── README.md (this documentation)
    └── sounds/
        ├── assignment_finished.mp3
        ├── request_fulfilled.mp3
        ├── task_complete.mp3
        ├── work_concluded.mp3
        └── work_finished.mp3
```

**Just 2 code files + 5 sound files = complete system!**

## Dependencies

- **Python**: 3.13+
- **pygame**: Audio playback library