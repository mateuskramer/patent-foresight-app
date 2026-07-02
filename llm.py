import os
import json
import time
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ─── Configuração ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"

# códigos de erro que justificam retry (rate limit / indisponibilidade)
_RETRYABLE = ("429", "503", "500", "quota", "RESOURCE_EXHAUSTED", "UNAVAILABLE")

_API_KEY = os.getenv("GEMINI_API_KEY", "")

# cliente é criado preguiçosamente para não quebrar a importação sem chave
_client: genai.Client | None = None

if not _API_KEY:
    logger.warning(
        "GEMINI_API_KEY não definida — as chamadas ao Gemini retornarão erro "
        "suave em runtime. Defina a variável no .env para habilitar a IA."
    )


def _get_client() -> genai.Client | None:
    """Retorna o cliente Gemini, criando-o sob demanda. None se faltar a chave."""
    global _client
    if not _API_KEY:
        return None
    if _client is None:
        _client = genai.Client(api_key=_API_KEY)
    return _client


def is_configured() -> bool:
    """True se há chave de API configurada."""
    return bool(_API_KEY)


# ─── Resultado tipado ─────────────────────────────────────────────────────────
class LLMResult:
    """
    Resultado de uma chamada ao LLM.

    - bool(result) e result.ok indicam sucesso.
    - str(result) devolve o texto (ou a mensagem de erro), então dá para usar
      diretamente onde antes se esperava uma string.
    """

    __slots__ = ("text", "ok", "error")

    def __init__(self, text: str = "", ok: bool = True, error: str | None = None):
        self.text = text
        self.ok = ok
        self.error = error

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return self.text if self.ok else (self.error or "")

    def __repr__(self) -> str:
        if self.ok:
            return f"LLMResult(ok=True, text={self.text[:50]!r}...)"
        return f"LLMResult(ok=False, error={self.error!r})"


def _make_config(
    temperature: float,
    max_output_tokens: int,
    system_instruction: str | None,
    response_mime_type: str | None,
    response_schema,
) -> types.GenerateContentConfig:
    kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


# ─── Geração de texto ─────────────────────────────────────────────────────────
def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_output_tokens: int = 4096,
    system_instruction: str | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    response_mime_type: str | None = None,
    response_schema=None,
) -> LLMResult:
    """
    Chama o Gemini com retry automático para erros transitórios (429/503/quota).

    Retorna um LLMResult. Em caso de falta de chave ou erro definitivo,
    result.ok é False e result.error contém a mensagem.
    """
    client = _get_client()
    if client is None:
        return LLMResult(ok=False, error="⚠️ GEMINI_API_KEY not set")

    config = _make_config(
        temperature, max_output_tokens,
        system_instruction, response_mime_type, response_schema,
    )

    last_err = "unknown error"
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            return LLMResult(text=text, ok=True)
        except Exception as e:
            last_err = str(e)
            transient = any(code in last_err for code in _RETRYABLE)
            if attempt < retries - 1 and transient:
                wait = backoff * (attempt + 1)
                logger.warning(
                    "Gemini erro transitório (tentativa %d/%d): %s — aguardando %.1fs",
                    attempt + 1, retries, last_err[:120], wait,
                )
                time.sleep(wait)
                continue
            logger.error("Gemini falhou: %s", last_err[:200])
            return LLMResult(ok=False, error=f"⚠️ Gemini error: {last_err}")

    return LLMResult(ok=False, error=f"⚠️ Gemini error: max retries exceeded ({last_err})")


# ─── Geração de JSON ──────────────────────────────────────────────────────────
def generate_json(
    prompt: str,
    *,
    schema=None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    system_instruction: str | None = None,
    retries: int = 3,
    backoff: float = 2.0,
):
    """
    Como generate(), mas força saída JSON e já faz o parse.

    - Se `schema` for fornecido (classe Pydantic ou dict de schema), o Gemini
      é instruído a respeitar a estrutura via response_schema — eliminando a
      necessidade de limpar ```json e tratar JSONDecodeError manualmente.
    - Sem schema, ainda força response_mime_type='application/json'.

    Retorna uma tupla (data, LLMResult):
      - data: objeto Python já parseado, ou None em caso de erro.
      - LLMResult: resultado bruto (útil para inspecionar erro/texto cru).

    Sempre inclui um fallback de limpeza de cercas de markdown caso o modelo
    devolva ```json apesar do mime_type.
    """
    result = generate(
        prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        retries=retries,
        backoff=backoff,
        response_mime_type="application/json",
        response_schema=schema,
    )

    if not result.ok:
        return None, result

    raw = result.text.strip()
    # fallback: remove cercas ```json ... ``` se vierem
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        return json.loads(raw), result
    except json.JSONDecodeError as e:
        logger.error("Falha ao parsear JSON do Gemini: %s | cru=%r", e, raw[:200])
        return None, LLMResult(ok=False, error=f"⚠️ JSON parse error: {e}")


# ─── Chat multi-turn ──────────────────────────────────────────────────────────
def create_chat(
    *,
    model: str = DEFAULT_MODEL,
    system_instruction: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 2000,
    history: list | None = None,
):
    """
    Cria uma sessão de chat (client.chats.create) com config padrão.

    Retorna o objeto chat do google-genai, ou None se faltar a chave.
    O chamador usa chat.send_message(texto).
    """
    client = _get_client()
    if client is None:
        return None

    config = _make_config(
        temperature, max_output_tokens,
        system_instruction, None, None,
    )
    return client.chats.create(
        model=model,
        history=history or [],
        config=config,
    )