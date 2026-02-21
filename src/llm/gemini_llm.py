import time
from google import genai
from google.genai.errors import ClientError
from src.config.config import GOOGLE_API_KEYS  # list of keys

# Build one client per key; rotate on rate-limit errors.
_clients = [genai.Client(api_key=k) for k in GOOGLE_API_KEYS]
_current_index = 0

# How long to wait (seconds) when ALL keys are exhausted before retrying.
# Free-tier quota resets per minute for RPM limits; use 60s as a safe default.
_EXHAUSTED_WAIT_SECONDS = 60


def _is_rate_limit(exc: ClientError) -> bool:
    """Return True if the ClientError is a 429 quota / rate-limit error."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    # Fallback: check string representation (the SDK embeds the HTTP code there)
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> dict:
    """
    Call the Gemini generate_content API with automatic key rotation and retry.

    Rotation strategy:
      1. Try the current key.
      2. On a 429 / RESOURCE_EXHAUSTED ClientError, advance to the next key
         and retry immediately.
      3. Once every key has been tried once, wait _EXHAUSTED_WAIT_SECONDS
         and do one final pass through all keys before giving up.
    """
    global _current_index
    last_exc = None
    n_keys = len(_clients)

    # Two passes: first pass rotates through all keys immediately;
    # second pass waits and tries all keys once more.
    for pass_num in range(2):
        for _ in range(n_keys):
            client = _clients[_current_index]
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                usage = response.usage_metadata
                return {
                    "text": response.text,
                    "tokens": {
                        "prompt": usage.prompt_token_count if usage else 0,
                        "response": usage.candidates_token_count if usage else 0,
                        "total": usage.total_token_count if usage else 0,
                    },
                    "key_index": _current_index,
                }
            except ClientError as e:
                if _is_rate_limit(e):
                    print(
                        f"[GeminiLLM] Key {_current_index + 1}/{n_keys} "
                        f"rate-limited (429). Rotating to next key…"
                    )
                    last_exc = e
                    _current_index = (_current_index + 1) % n_keys
                else:
                    # Non-rate-limit error (e.g. invalid request) — don't retry
                    raise

        # All keys exhausted on this pass — wait before the second pass
        if pass_num == 0:
            print(
                f"[GeminiLLM] All {n_keys} key(s) exhausted. "
                f"Waiting {_EXHAUSTED_WAIT_SECONDS}s before retrying…"
            )
            time.sleep(_EXHAUSTED_WAIT_SECONDS)

    raise RuntimeError(
        f"All {n_keys} Gemini API key(s) are rate-limited after retrying. "
        "Consider adding more keys (GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, …) "
        "to your .env file, or upgrade to a paid Gemini API tier."
    ) from last_exc