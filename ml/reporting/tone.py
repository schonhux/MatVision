TONE_INSTRUCTIONS = {
    "balanced": (
        "Use a calm, direct coaching voice. Recognize what worked, explain the main issue, "
        "and finish with one practical priority."
    ),
    "hard": (
        "Use blunt, concise accountability. Call out avoidable mistakes directly and use short "
        "commands, while keeping every criticism tied to cited match evidence."
    ),
    "extreme": (
        "Use high-intensity locker-room coaching and rhetorical challenge. Be forceful and urgent, "
        "but never insult the athlete's identity or worth, invent a technique, diagnose an injury, "
        "or make a claim without cited evidence."
    ),
}


def coach_tone_instruction(tone: str) -> str:
    try:
        return TONE_INSTRUCTIONS[tone]
    except KeyError as exc:
        raise ValueError(f"Unknown coach tone: {tone}") from exc
