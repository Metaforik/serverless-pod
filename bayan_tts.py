import os

# ============================================================
# HUGGING FACE CACHE CONFIGURATION
# ============================================================
#
# If a RunPod network volume is mounted at /runpod-volume,
# Hugging Face models will be cached there.
#
# This is important because the Bayan model is approximately
# 7.57 GB. We do NOT want to download it on every worker start.
#
# If /runpod-volume does not exist, we fall back to /tmp.
#

RUNPOD_VOLUME = "/runpod-volume"

if os.path.isdir(RUNPOD_VOLUME):
    HF_CACHE_DIR = os.path.join(
        RUNPOD_VOLUME,
        "huggingface"
    )
else:
    HF_CACHE_DIR = "/tmp/huggingface"

os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Tell Hugging Face libraries where to keep their cache.
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(
    HF_CACHE_DIR,
    "hub"
)
os.environ["TRANSFORMERS_CACHE"] = os.path.join(
    HF_CACHE_DIR,
    "transformers"
)

print("=" * 60)
print("Hugging Face cache configuration")
print(f"HF cache: {HF_CACHE_DIR}")
print("=" * 60)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch
import soundfile as sf

from huggingface_hub import snapshot_download
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_ID = "Party-Lemur/bayan"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ORPHEUS SPECIAL TOKENS
# ============================================================

SOH_TOKEN = 128259
EOT_TOKEN = 128009
EOH_TOKEN = 128260
SOS_TOKEN = 128257
EOS_TOKEN = 128258
OFFSET = 128266


# ============================================================
# GENERATION DEFAULTS
# ============================================================

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_MAX_NEW_TOKENS = 500


# ============================================================
# TTS GENERATOR
# ============================================================

class BayanTTS:

    def __init__(self):

        print("=" * 60)
        print("Initializing Bayan TTS")
        print(f"Model: {MODEL_ID}")
        print(f"Device: {DEVICE}")
        print(f"HF cache: {HF_CACHE_DIR}")
        print("=" * 60)

        if DEVICE != "cuda":
            raise RuntimeError(
                "CUDA is not available. "
                "This worker requires an NVIDIA GPU."
            )

        # ----------------------------------------------------
        # Hugging Face authentication
        # ----------------------------------------------------

        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Please add your Hugging Face Read token "
                "to the RunPod endpoint environment variables."
            )

        print("Hugging Face authentication token detected.")

        # ----------------------------------------------------
        # Load SNAC
        # ----------------------------------------------------

        print("Loading SNAC...")

        snac_path = self._get_cached_model(
            repo_id=SNAC_MODEL_ID,
            token=None,
            name="SNAC"
        )

        self.snac_model = (
            SNAC.from_pretrained(snac_path)
            .to(DEVICE)
            .eval()
        )

        print("SNAC loaded successfully.")

        # ----------------------------------------------------
        # Load Bayan model
        # ----------------------------------------------------

        print("Preparing Bayan model cache...")

        bayan_path = self._get_cached_model(
            repo_id=MODEL_ID,
            token=hf_token,
            name="Bayan"
        )

        print(f"Bayan model path: {bayan_path}")

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            bayan_path,
            local_files_only=True,
        )

        print("Tokenizer loaded successfully.")

        # ----------------------------------------------------
        # Load fine-tuned model
        # ----------------------------------------------------

        print("Loading Bayan model into GPU...")

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                bayan_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(DEVICE)
            .eval()
        )

        print("Bayan model loaded successfully.")
        print("=" * 60)


    # ========================================================
    # HUGGING FACE MODEL CACHE
    # ========================================================

    def _get_cached_model(
        self,
        repo_id,
        token=None,
        name="model",
    ):
        """
        Download a Hugging Face repository once and reuse the
        cached snapshot on subsequent worker starts.

        If /runpod-volume exists, the cache survives worker
        replacement/restarts provided the RunPod network volume
        remains attached to the endpoint.

        snapshot_download() automatically reuses an existing
        cached snapshot instead of downloading everything again.
        """

        print("-" * 60)
        print(f"Checking {name} cache")
        print(f"Repository: {repo_id}")
        print(f"Cache directory: {HF_CACHE_DIR}")

        try:

            snapshot_path = snapshot_download(
                repo_id=repo_id,
                token=token,
                cache_dir=HF_CACHE_DIR,
            )

            print(f"{name} snapshot ready.")
            print(f"Local path: {snapshot_path}")

            return snapshot_path

        except Exception as e:

            print("=" * 60)
            print(f"ERROR while downloading/caching {name}")
            print(f"Repository: {repo_id}")
            print(f"Error: {e}")
            print("=" * 60)

            raise


    # ========================================================
    # SNAC AUDIO DECODING
    # ========================================================

    def decode_audio(self, code_list):

        """
        Convert Orpheus interleaved audio tokens into
        SNAC codes and decode to 24 kHz audio.
        """

        layer_1 = []
        layer_2 = []
        layer_3 = []

        for i in range(len(code_list) // 7):

            layer_1.append(
                np.clip(
                    code_list[7 * i],
                    0,
                    4095
                )
            )

            layer_2.append(
                np.clip(
                    code_list[7 * i + 1] - 4096,
                    0,
                    4095
                )
            )

            layer_3.append(
                np.clip(
                    code_list[7 * i + 2] - (2 * 4096),
                    0,
                    4095
                )
            )

            layer_3.append(
                np.clip(
                    code_list[7 * i + 3] - (3 * 4096),
                    0,
                    4095
                )
            )

            layer_2.append(
                np.clip(
                    code_list[7 * i + 4] - (4 * 4096),
                    0,
                    4095
                )
            )

            layer_3.append(
                np.clip(
                    code_list[7 * i + 5] - (5 * 4096),
                    0,
                    4095
                )
            )

            layer_3.append(
                np.clip(
                    code_list[7 * i + 6] - (6 * 4096),
                    0,
                    4095
                )
            )

        codes = [
            torch.tensor(
                layer_1,
                dtype=torch.long
            ).unsqueeze(0).to(DEVICE),

            torch.tensor(
                layer_2,
                dtype=torch.long
            ).unsqueeze(0).to(DEVICE),

            torch.tensor(
                layer_3,
                dtype=torch.long
            ).unsqueeze(0).to(DEVICE),
        ]

        with torch.no_grad():

            audio = self.snac_model.decode(codes)

        return audio.cpu().numpy().squeeze()


    # ========================================================
    # SENTENCE GENERATION
    # ========================================================

    def generate_sentence(
        self,
        text,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    ):

        text = text.strip()

        if not text:
            raise ValueError(
                "Text cannot be empty."
            )

        # ----------------------------------------------------
        # Tokenize text
        # ----------------------------------------------------

        input_ids = self.tokenizer(
            text,
            return_tensors="pt"
        ).input_ids

        # ----------------------------------------------------
        # Orpheus prompt structure
        # ----------------------------------------------------

        start_token = torch.tensor(
            [[SOH_TOKEN]],
            dtype=torch.long
        )

        end_tokens = torch.tensor(
            [[EOT_TOKEN, EOH_TOKEN]],
            dtype=torch.long
        )

        full_input_ids = torch.cat(
            [
                start_token,
                input_ids,
                end_tokens
            ],
            dim=1
        ).to(DEVICE)

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        print(f"Generating: {text}")

        with torch.no_grad():

            generated_ids = self.model.generate(
                input_ids=full_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=EOS_TOKEN,
            )

        # ----------------------------------------------------
        # Find speech start token
        # ----------------------------------------------------

        token_indices = (
            generated_ids == SOS_TOKEN
        ).nonzero(as_tuple=True)

        if len(token_indices[1]) == 0:

            raise RuntimeError(
                "SOS token was not found in model output."
            )

        last_idx = token_indices[1][-1].item()

        speech_tokens = generated_ids[
            0,
            last_idx + 1:
        ]

        # ----------------------------------------------------
        # Remove EOS
        # ----------------------------------------------------

        speech_tokens = speech_tokens[
            speech_tokens != EOS_TOKEN
        ].tolist()

        # ----------------------------------------------------
        # Make length divisible by 7
        # ----------------------------------------------------

        trimmed_len = (
            len(speech_tokens) // 7
        ) * 7

        speech_tokens = speech_tokens[
            :trimmed_len
        ]

        if not speech_tokens:

            raise RuntimeError(
                "No speech tokens were generated."
            )

        # ----------------------------------------------------
        # Remove Orpheus token offset
        # ----------------------------------------------------

        speech_tokens = [
            token - OFFSET
            for token in speech_tokens
        ]

        # ----------------------------------------------------
        # Decode with SNAC
        # ----------------------------------------------------

        audio = self.decode_audio(
            speech_tokens
        )

        return audio


    # ========================================================
    # LONG-FORM GENERATION
    # ========================================================

    def generate(
        self,
        text,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    ):

        import re

        clean_text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if not clean_text:

            raise ValueError(
                "Text cannot be empty."
            )

        sentences = re.split(
            r"(?<=[.!?]) +",
            clean_text
        )

        print(
            f"Processing {len(sentences)} sentence(s)."
        )

        combined_audio = []

        # 0.4 seconds of silence.

        silence = np.zeros(
            int(24000 * 0.4),
            dtype=np.float32
        )

        for i, sentence in enumerate(sentences):

            sentence = sentence.strip()

            if len(sentence) < 2:
                continue

            print(
                f"Sentence {i + 1}/{len(sentences)}"
            )

            audio_chunk = self.generate_sentence(
                sentence,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )

            combined_audio.append(
                audio_chunk
            )

            combined_audio.append(
                silence
            )

        if not combined_audio:

            raise RuntimeError(
                "No audio was generated."
            )

        final_audio = np.concatenate(
            combined_audio
        )

        return final_audio


# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

generator = None


def get_generator():

    global generator

    if generator is None:

        print("=" * 60)
        print("Creating BayanTTS generator for this worker.")
        print("This should happen only once per worker.")
        print("=" * 60)

        generator = BayanTTS()

        print("=" * 60)
        print("BayanTTS generator is now cached in memory.")
        print("=" * 60)

    else:

        print(
            "Reusing cached BayanTTS generator."
        )

    return generator


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    tts = get_generator()

    audio = tts.generate(
        "Hello, this is a test of my fine tuned Bayan voice.",
        max_new_tokens=500,
    )

    output_file = "bayan_test.wav"

    sf.write(
        output_file,
        audio,
        24000
    )

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print(f"Saved: {output_file}")
    print("=" * 60)
