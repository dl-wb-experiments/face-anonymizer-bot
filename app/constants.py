from enum import IntEnum, Enum


class BotConversationState(IntEnum):
    start = 1
    select_anonymization_type = 2
    upload_media = 3


class AnonymizeType(Enum):
    blur = 'blur'
    emoji = 'emoji'


class Emotion(Enum):
    anger = 'anger'
    happy = 'happy'
    neutral = 'neutral'
    sad = 'sad'
    surprise = 'surprise'
