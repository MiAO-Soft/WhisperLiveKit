import logging
import torch
import time
from .timed_text import TimedText
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Union
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PUNCTUATION_MARKS = {".", "!", "?", "。", "！", "？"}


class OpenAITranslationBackend:
    def __init__(
        self,
        source_lang,
        target_lang,
        base_url=None,
        model_name=None,
        verbose: bool = False,
        max_kept_sentences: int = 1,
    ):
        self.base_url = base_url
        self.verbose = verbose
        self.max_kept_sentences = max_kept_sentences

        self.translator = None

        self.source_lang = source_lang
        self.target_lang = target_lang

        self.sentence_end_token_ids = set()

        self.input_buffer = []
        self.last_translation = ''
        self.previous_tokens = []
        self.stable_prefix_segments = []
        self.stable_prefix_tokens = torch.tensor([], dtype=torch.int64)
        self.n_remaining_input_punctuation = 0

        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = "sk-fake"
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def reset(self, duration: float):
        self.last_translation = ''
        self.input_buffer = []
        self.target_prefix_tokens = []
        self.previous_tokens = []
        self.stable_prefix_segments = []
        self.stable_prefix_tokens = torch.tensor([], dtype=torch.int64)
        self.n_remaining_input_punctuation = 0

    def create_translation_prompt(self, text, from_lan, to_lan) -> str:
        prompt = f"""Translate following text from {from_lan} to {to_lan}, be aware that the translation u translated last time is '{self.last_translation}', please dont modify the translated part:
{text}
"""
        return prompt
    
    def _trim(self) -> bool:
        x = 5
        # print(f'Before trimming: {len(self.input_buffer)} {len(self.stable_prefix_segments)}')
        self.input_buffer = self.input_buffer[-x - 1 :]
        self.stable_prefix_segments = self.stable_prefix_segments[-x:]
        # print(f'After trimming: {len(self.input_buffer)} {len(self.stable_prefix_segments)}')

    def simple_translation(self, text, from_lan, to_lan):
        if self.verbose:
            start_time = time.time()
        response = self.client.chat.completions.create(
            model="fake",
            messages=[
                {"role": "system", "content": "You are a professional translation engine, please translate the text into a colloquial, professional, elegant and fluent content, without the style of machine translation. You must only translate the text content, never interpret it."},
                {"role": "user", "content": self.create_translation_prompt(text, from_lan, to_lan)}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        if self.verbose:
            response_time = time.time() - start_time
            print(
                f"api response time: {response_time:.4f}s"
            )

        result = response.choices[0].message.content.strip()

        # logger.info(f"translated 【{text}】\n<--------------->\n【{result}】")

        return result

    def handle_input_sentences(self, buffer_text):
        early_cut = False
        last_punct_pos = -1

        for i, char in enumerate(buffer_text):
            if char in PUNCTUATION_MARKS:
                last_punct_pos = i

        if last_punct_pos >= 0:
            text_to_process = buffer_text[: last_punct_pos + 1]
            remaining_text = buffer_text[last_punct_pos + 1 :]
            char_count = 0
            split_index = len(self.input_buffer)
            for idx, timed_text in enumerate(self.input_buffer):
                char_count += len(timed_text.text)
                if char_count > last_punct_pos:
                    split_index = idx
                    if char_count - len(timed_text.text) <= last_punct_pos:
                        chars_before = (
                            last_punct_pos + 1 - (char_count - len(timed_text.text))
                        )
                        before_text = timed_text.text[:chars_before]
                        after_text = timed_text.text[chars_before:]
                        if after_text:
                            self.input_buffer = [
                                TimedText(
                                    text=after_text,
                                    start=timed_text.start,
                                    end=timed_text.end,
                                )
                            ] + self.input_buffer[idx + 1 :]
                        else:
                            self.input_buffer = self.input_buffer[idx + 1 :]
                    break

            buffer_text = text_to_process
            early_cut = True
            print(
                f'\033[33mEarly cut. Processing: "{text_to_process}" | Remaining: "{remaining_text}"\033[0m'
            )

        return buffer_text, early_cut

    def translate(self, text: Optional[str | TimedText] = None) -> str:
        if type(text) == str:
            self.input_buffer.append(TimedText(text))
        elif type(text) == TimedText:
            self.input_buffer.append(text)

        # self._trim()
        buffer_text = "".join([token.text for token in self.input_buffer])
        from_language = self.input_buffer[-1].detected_language
        to_language = self.target_lang if from_language == self.source_lang else self.source_lang
        # buffer_text, early_cut = self.handle_input_sentences(buffer_text)
        word_count = len(buffer_text)
        if word_count < 3:
            return ""
        return self.simple_translation(buffer_text, from_language, to_language)


if __name__ == "__main__":
    import argparse
    from nllw.test_strings import *
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Quick translation backend smoke test."
    )
    parser.add_argument(
        "--backend", choices=["transformers", "ctranslate2"], default="ctranslate2"
    )
    parser.add_argument("--nllb-size", default="600M")
    parser.add_argument("--source-lang", default="fra_Latn")
    parser.add_argument("--target-lang", default="eng_Latn")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--verbose", default="True")
    args = parser.parse_args()

    translation_model = load_model(
        [args.source_lang], nllb_backend=args.backend, nllb_size=args.nllb_size
    )
    source_lang_nllb = convert_to_nllb_code(args.source_lang) or args.source_lang
    target_lang_nllb = convert_to_nllb_code(args.target_lang) or args.target_lang
    translation_backend = TranslationBackend(
        source_lang=source_lang_nllb,
        target_lang=target_lang_nllb,
        model_name=translation_model.model_name,
        model=translation_model.translator,
        tokenizer=translation_model.get_tokenizer(source_lang_nllb),
        backend_type=translation_model.backend_type,
        verbose=args.verbose,
        ctranslate2_beam_size=args.beam_size,
    )

    src_texts = src_2_fr
    l_vals_with_cache = []
    for i in range(0, len(src_texts)):
        truncated_text = src_texts[i]
        print(f"\n\n{i}/{len(src_texts) + 1}: {truncated_text}")
        stable_translation, buffer = translation_backend.translate(truncated_text)

        full_output = stable_translation + buffer
        l_vals_with_cache.append(
            {
                "input": truncated_text,
                "stable_translation": stable_translation,
                "stable_prefix_tokens": translation_backend.stable_prefix_tokens,
                "input_word_count": len(truncated_text.split()),
                "stable_word_count": (
                    len(stable_translation.split()) if stable_translation else 0
                ),
                "total_output_word_count": (
                    len(full_output.split()) if full_output else 0
                ),
                "backend": args.backend,
            }
        )
    pd.DataFrame(l_vals_with_cache).to_pickle("export_with_tokens.pkl")
