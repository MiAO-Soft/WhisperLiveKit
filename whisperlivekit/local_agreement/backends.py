import sys
import logging
import io
import soundfile as sf
import math
from typing import List
import numpy as np
import requests, time
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.model_paths import resolve_model_path, model_path_and_type
from whisperlivekit.whisper.transcribe import transcribe as whisper_transcribe
logger = logging.getLogger(__name__)
class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperASR(ASRBase):
    """Uses WhisperLiveKit's built-in Whisper implementation."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from whisperlivekit.whisper import load_model as load_model

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            if resolved_path.is_dir():
                pytorch_path, _, _ = model_path_and_type(resolved_path)
                if pytorch_path is None:
                    raise FileNotFoundError(
                        f"No supported PyTorch checkpoint found under {resolved_path}"
                    )
                resolved_path = pytorch_path
            logger.debug(f"Loading Whisper model from custom path {resolved_path}")
            return load_model(str(resolved_path))

        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperASR")

        return load_model(model_size, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        options = dict(self.transcribe_kargs)
        options.pop("vad", None)
        options.pop("vad_filter", None)
        language = self.original_language if self.original_language else None

        result = whisper_transcribe(
            self.model,
            audio,
            language=language,
            initial_prompt=init_prompt,
            condition_on_previous_text=True,
            word_timestamps=True,
            **options,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the Whisper result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(
                    word["start"],
                    word["end"],
                    word["word"],
                    probability=word.get("probability"),
                )
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        logger.warning("VAD is not currently supported for WhisperASR backend and will be ignored.")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading faster-whisper model from {resolved_path}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
                              

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading MLX Whisper model from {resolved_path}. model_size parameter is not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                probability=word["probability"]
                token = ASRToken(word["start"], word["end"], word["word"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.direct_english_translation = False

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if not self.direct_english_translation and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True

class WhisperCppApiASR(ASRBase):
    """Uses WhisperCpp's API for audio transcription."""

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr, whisper_server_port=18181):
        self.logfile = logfile

        self.model_name = "ggml-medium"
        self.original_language = None if lan == "auto" else lan # ISO-639-1 language code
        self.response_format = "verbose_json" 
        self.temperature = temperature
        self.whisper_server_port = whisper_server_port

        self.load_model()

        self.use_vad_opt = False

        self.base_url = f"http://localhost:{self.whisper_server_port}"

        # reset the task in set_translate_task
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        import os
        import sys
        import platform
        import subprocess
        import time
        import requests
        
        system = platform.system().lower()
        
        whisper_cpp_dir = os.path.join(os.path.dirname(__file__), "../../whisper.cpp")
        
        if system == "windows":
            server_executable = os.path.join(whisper_cpp_dir, "win", "whisper-server.exe")
        elif system == "linux":
            server_executable = os.path.join(whisper_cpp_dir, "linux", "whisper-server")
        else:
            raise NotImplementedError(f"Unsupported operating system: {system}")
        
        if not os.path.exists(server_executable):
            raise FileNotFoundError(f"whisper-server not found at: {server_executable}")
        
        model_path = os.path.join(whisper_cpp_dir, "models", self.model_name + ".bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        vad_path = os.path.join(whisper_cpp_dir, "models", "ggml-silero-v6.2.0.bin")
        if not os.path.exists(vad_path):
            raise FileNotFoundError(f"Model file not found at: {vad_path}")
        
        public_path = os.path.join(whisper_cpp_dir, "static")
        
        command_args = [
            server_executable,
            "-m", model_path,
            "--host", "0.0.0.0",
            "--port", str(self.whisper_server_port),
            "-t", "8",
            "-l", "auto",
            "--public", public_path,
            "-p", "4",
            "--vad",
            "--vad-model", vad_path,
            "-mc", "0",
            "-di"
        ]
        
        try:
            try:
                health_url = f"http://127.0.0.1:{self.whisper_server_port}/health"
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"whisper-server is already running on port {self.whisper_server_port}")
                    self.server_process = None
                else:
                    raise requests.ConnectionError("Server not responding properly")
            except (requests.ConnectionError, requests.Timeout):
                logger.info(f"Starting whisper-server with command: {' '.join(command_args)}")
                self.server_process = subprocess.Popen(
                    command_args,
                    cwd=os.path.dirname(server_executable),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                logger.info(f"Waiting for whisper-server to start on port {self.whisper_server_port}...")
                for i in range(30):  # 最多等待30秒
                    try:
                        health_url = f"http://127.0.0.1:{self.whisper_server_port}/health"
                        response = requests.get(health_url, timeout=2)
                        if response.status_code == 200:
                            logger.info(f"whisper-server started successfully on port {self.whisper_server_port}")
                            break
                    except (requests.ConnectionError, requests.Timeout):
                        if i == 29: 
                            raise TimeoutError(f"whisper-server failed to start within 30 seconds on port {self.whisper_server_port}")
                        time.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to start whisper-server: {e}")
            if hasattr(self, 'server_process') and self.server_process:
                self.server_process.terminate()
            raise
        
        # from openai import OpenAI
        # self.client = OpenAI(
        #     base_url=f"http://127.0.0.1:{self.whisper_server_port}/v1",
        #     api_key="sk-not-needed"
        # )

        self.transcribed_seconds = 0  # for logging how many seconds were processed by API, to know the cost
    
    def ts_words(self, res):
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in res.segments:
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o = []
        if res is None:
            return o
        for segment in res.segments:
            for word in segment['words']:
                start = word['start']
                end = word['end']
                if any(s[0] <= start <= s[1] for s in no_speech_segments):
                    # print("Skipping word", word.get("word"), "because it's in a no-speech segment")
                    continue
                o.append(ASRToken(start, end, word['word']))
        return o


    def segments_end_ts(self, res):
        o = []
        for segment in res.segments:
            o.append(segment['end'])
        return o

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        from openai.types.audio import Transcription

        # Write the audio data to a buffer
        buffer = io.BytesIO()
        buffer.name = "temp1.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)  # Reset buffer's position to the beginning

        self.transcribed_seconds += math.ceil(len(audio_data)/16000)  # it rounds up to the whole seconds

        files = {
            'file': ('temp1.wav', buffer, 'audio/wav')
        }

        data = {
            "model": self.model_name,
            "response_format": self.response_format,
            "temperature": str(self.temperature),
            "temperature_inc": "0.2", 
            "timestamp_granularities[]": ["word", "segment"],
            "vad": "false",
            "n_processors": "1"
        }

        if self.task != "translate" and self.original_language:
            data["language"] = self.original_language
        if prompt:
            data["prompt"] = prompt

        start_time = time.time()

        resp = requests.post(
            f"{self.base_url}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        resp.raise_for_status()

        elapsed = time.time() - start_time
        print(f"[Transcribe API] 响应时间: {elapsed:.3f} 秒")

        json_data = resp.json()
        transcription = Transcription.model_validate(json_data)
        return transcription

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"
