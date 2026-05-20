"""
Async API call utility — OpenAI-compatible, with multimodal support and 429 retry.
"""
import asyncio
import aiohttp
import base64
import os
from typing import Optional

from config import API_KEY, API_URL, API_MODEL, API_TIMEOUT, TEMPERATURE, MAX_TOKENS, REQUEST_DELAY

_image_cache: dict = {}

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3  # seconds; doubles on each retry (exponential back-off)


def _encode_image(image_path: str) -> str:
    """Base64-encode an image file (result is cached in memory)."""
    if not image_path or not os.path.exists(image_path):
        return ""

    if image_path in _image_cache:
        return _image_cache[image_path]

    try:
        with open(image_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")
            _image_cache[image_path] = encoded
            return encoded
    except Exception as exc:
        print(f"[api_utils] Image encoding failed for {image_path}: {exc}")
        return ""


async def fetch_api(
    session: aiohttp.ClientSession,
    prompt: str,
    image_path: Optional[str] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    timeout: int = API_TIMEOUT,
    request_delay: float = REQUEST_DELAY,
) -> str:
    """
    Send a (optionally multimodal) prompt to the configured LLM endpoint.

    Supports text-only and text+image (base64 inline) requests in the
    OpenAI chat-completion format.  Automatically retries on HTTP 429
    with exponential back-off.

    Returns the model's reply string, or an error message string on failure.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if image_path and os.path.exists(image_path):
        image_data = _encode_image(image_path)
        content = (
            [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
            if image_data
            else prompt
        )
    else:
        content = prompt

    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            await asyncio.sleep(request_delay)
            async with session.post(
                API_URL, headers=headers, json=payload, timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"].strip()
                    return f"Unexpected API response format: {str(result)[:200]}"

                if response.status == 429:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    return f"API rate limited (429), failed after {MAX_RETRIES} retries"

                if response.status == 401:
                    return "API authentication failed (401): check MCTL_API_KEY"

                error_text = await response.text()
                return f"API request failed (status {response.status}): {error_text[:200]}"

        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(wait)
                continue
            return f"API call timed out (>{timeout}s), failed after {MAX_RETRIES} retries"

        except Exception as exc:
            return f"API call exception: {str(exc)[:200]}"

    return "API call failed: exceeded max retries"
