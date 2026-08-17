print("handler.py loaded v0.2")


import base64
import io
import time

import numpy as np
import runpod
import soundfile as sf

from bayan_tts import get_generator


# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================
#
# The model is NOT loaded when this module is imported.
#
# It will be loaded on the first request and then reused for
# all subsequent requests handled by this worker.
#
# This prevents the 7.57 GB model from being loaded once per
# request.
# ============================================================

generator = None


# ============================================================
# AUDIO ENCODING
# ============================================================

def audio_to_base64(audio, sample_rate=24000):
    """
    Convert a NumPy audio array into a WAV file and then
    encode the WAV as base64 for the RunPod response.
    """

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    """
    RunPod Serverless handler.

    Expected input:

    {
        "input": {
            "text": "Hello, this is a test.",
            "temperature": 0.6,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "max_new_tokens": 500
        }
    }
    """

    global generator

    # --------------------------------------------------------
    # Read input
    # --------------------------------------------------------

    job_input = job.get("input", {})

    text = job_input.get("text")

    if not text:
        return {
            "error": "No text supplied."
        }

    text = str(text).strip()

    if not text:
        return {
            "error": "Text is empty."
        }

    # --------------------------------------------------------
    # Generation parameters
    # --------------------------------------------------------

    temperature = float(
        job_input.get(
            "temperature",
            0.6
        )
    )

    top_p = float(
        job_input.get(
            "top_p",
            0.95
        )
    )

    repetition_penalty = float(
        job_input.get(
            "repetition_penalty",
            1.1
        )
    )

    max_new_tokens = int(
        job_input.get(
            "max_new_tokens",
            500
        )
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if max_new_tokens < 1:
        return {
            "error": "max_new_tokens must be greater than 0."
        }

    if temperature <= 0:
        return {
            "error": "temperature must be greater than 0."
        }

    if not 0 < top_p <= 1:
        return {
            "error": "top_p must be between 0 and 1."
        }

    if repetition_penalty <= 0:
        return {
            "error": "repetition_penalty must be greater than 0."
        }

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print("=" * 60)
    print("New Bayan TTS request")
    print("=" * 60)

    print(f"Text: {text}")
    print(f"Temperature: {temperature}")
    print(f"Top P: {top_p}")
    print(f"Repetition penalty: {repetition_penalty}")
    print(f"Max new tokens: {max_new_tokens}")

    # --------------------------------------------------------
    # Load model ONCE per worker
    # --------------------------------------------------------
    #
    # The first request loads:
    #
    #   Party-Lemur/bayan
    #   SNAC
    #   tokenizer
    #
    # Subsequent requests reuse the same generator.
    # --------------------------------------------------------

    if generator is None:

        print("=" * 60)
        print("First request received.")
        print("Loading Bayan TTS model...")
        print("=" * 60)

        generator = get_generator()

        print("=" * 60)
        print("Bayan TTS model is ready.")
        print("=" * 60)

    else:

        print("Reusing already-loaded Bayan TTS model.")

    # --------------------------------------------------------
    # Generate audio
    # --------------------------------------------------------

    start_time = time.time()

    try:

        audio = generator.generate(
            text=text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

    except Exception as e:

        print("=" * 60)
        print("GENERATION ERROR")
        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }

    # --------------------------------------------------------
    # Calculate timing
    # --------------------------------------------------------

    generation_time = (
        time.time() - start_time
    )

    audio_duration = (
        len(audio) / 24000
    )

    # --------------------------------------------------------
    # Convert WAV to base64
    # --------------------------------------------------------

    try:

        audio_base64 = audio_to_base64(
            audio,
            sample_rate=24000,
        )

    except Exception as e:

        print("=" * 60)
        print("AUDIO ENCODING ERROR")
        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }

    # --------------------------------------------------------
    # Final logging
    # --------------------------------------------------------

    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Generation time: "
        f"{generation_time:.2f} seconds"
    )

    print(
        f"Audio duration: "
        f"{audio_duration:.2f} seconds"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "audio_base64": audio_base64,
        "sample_rate": 24000,
        "audio_duration": audio_duration,
        "generation_time": generation_time,
    }


# ============================================================
# RUNPOD SERVERLESS START
# ============================================================
#
# IMPORTANT:
#
# This is intentionally at MODULE LEVEL rather than inside:
#
#     if __name__ == "__main__":
#
# RunPod's repository scanner can therefore directly discover:
#
#     runpod.serverless.start(...)
#
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
