import requests

DEEPL_API_KEY = "fa9dad85-d2d5-4b0c-83dc-2fa076f32753:fx"

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    response = requests.post(url, headers=headers, data=data, timeout=12)
    response.raise_for_status()
    result = response.json()
    return result["translations"][0]["text"]
