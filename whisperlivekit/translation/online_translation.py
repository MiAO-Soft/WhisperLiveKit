import logging
import torch
from .timed_text import TimedText

from .backends import OpenAITranslationBackend

"""
Interface for WhisperLiveKit. For other usages, it may be wiser to look at nllw.core directly.
"""

logger = logging.getLogger(__name__)
MIN_SILENCE_DURATION_DEL_BUFFER = 1.0

def load_model():
    pass

class OnlineTranslation:
    def __init__(self, openai_base_url, input_languages: list, output_languages: list):
        self.input_languages = input_languages
        self.output_languages = output_languages

        self.last_buffer = TimedText()
        self.last_end_time: float = 0.0 

        self.backend = OpenAITranslationBackend(
            source_lang=self.input_languages[0],
            target_lang=self.output_languages[0],
            base_url=openai_base_url
        )

    def insert_tokens(self, tokens):
        self.backend.input_buffer.extend(tokens)
    
    def process(self):
        if self.backend.input_buffer:
            start_time = self.backend.input_buffer[0].start
            end_time = self.backend.input_buffer[-1].end
        else:
            start_time = end_time = 0.0
        buffer_text = self.backend.translate()
        new_validated_translation = TimedText(
            text=buffer_text,
            start=start_time,
            end=end_time
        )

        buffer = TimedText(
            text='',
            start=start_time,
            end=end_time
        )
        self.last_buffer = buffer
        return new_validated_translation, buffer

    def validate_buffer_and_reset(self, duration: float = None):
        self.backend.reset(duration)
        
        return self.last_buffer, TimedText()

    def insert_silence(self, duration: float = None):
        if duration >= 5.0:
            self.validate_buffer_and_reset()