import os
import re
import logging
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

openai_api_key = "SUA-APIKEY-DEEPSEEK"

client = OpenAI(api_key=openai_api_key, base_url="https://api.deepseek.com")

MODEL = "deepseek-chat"

MAX_CHARS = 400  # corte final da fala (conta só o texto falado, não as tags)
MAX_TOKENS = 250  # teto de geração; folga extra para as tags <lang>
MAX_TURNS = 5  # pares pergunta/resposta mantidos no contexto

# Locales aceitos pela Alexa no atributo xml:lang da tag <lang>.
SSML_LOCALES = (
    "de-DE|en-AU|en-CA|en-GB|en-IN|en-US|es-ES|es-MX|es-US|"
    "fr-CA|fr-FR|hi-IN|it-IT|ja-JP|pt-BR"
)
LANG_TOKEN = re.compile(
    r'<lang\s+xml:lang=["\'](?P<loc>' + SSML_LOCALES + r')["\']\s*>'
    r"|(?P<close></lang\s*>)",
    re.IGNORECASE,
)
ANY_TAG = re.compile(r"<[^>]*>")  # qualquer outra tag é descartada
MARKDOWN = re.compile(r"[*_`#~|]+")

SYSTEM_PROMPT = (
    "Você é o assistente pessoal do Rafael e responde por voz na Alexa, "
    "em Português do Brasil. "
    "Fale de forma natural e direta, como numa conversa, e vá direto ao ponto: "
    "primeira frase já responde a pergunta. "
    "Máximo de 400 caracteres, em texto corrido. "
    "Nunca use listas, markdown, emojis, asteriscos, títulos, tabelas ou código. "
    "PRONÚNCIA: envolva nomes próprios, marcas e termos técnicos estrangeiros na "
    'tag <lang xml:lang="en-US">assim</lang>, trocando o código pelo idioma de '
    "origem (en-US, es-ES, fr-FR, it-IT, de-DE, ja-JP). "
    'Exemplo: Os pioneiros foram <lang xml:lang="en-US">Grace Hopper</lang> e '
    '<lang xml:lang="en-US">Alan Turing</lang>. '
    "Não use nenhuma outra tag além de <lang>. "
    "Não marque palavras já incorporadas ao português, como software, site ou mouse. "
    "Escreva números, símbolos e siglas por extenso quando isso soar melhor falado. "
    "Se a pergunta for ambígua, responda a interpretação mais provável e ofereça detalhar. "
    "Se não souber, diga em uma frase e sugira o próximo passo. "
    "Nada de saudações, desculpas ou frases de preenchimento."
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = (
            "Bem vindo ao assistente 'dipi siqui'! Qual a sua pergunta?"
        )

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class GptQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slots = handler_input.request_envelope.request.intent.slots
        query = slots["query"].value if slots and "query" in slots else None

        if not query or len(query.strip()) < 2:
            speak_output = "Não peguei a pergunta. Pode repetir?"
            return (
                handler_input.response_builder.speak(speak_output)
                .ask(speak_output)
                .response
            )

        response = generate_gpt_response(query.strip())

        return (
            handler_input.response_builder.speak(response)
            .ask("Pode perguntar outra coisa ou falar: sair.")
            .response
        )


def trim_history():
    """Mantém o system prompt e só os últimos MAX_TURNS pares de mensagens."""
    global messages
    limite = MAX_TURNS * 2
    if len(messages) > limite + 1:
        messages = messages[:1] + messages[-limite:]


def _norm_locale(loc):
    """en-us / EN-US -> en-US."""
    return loc[:2].lower() + "-" + loc[3:].upper()


def _segments(text):
    """Quebra a resposta em [(trecho, locale ou None)] a partir das tags <lang>.

    Só a tag <lang> com locale válido é reconhecida. Qualquer outra tag que o
    modelo invente é removida, então nunca chega marcação estranha no SSML.
    """
    brutos = []
    lang = None
    fim = 0

    for m in LANG_TOKEN.finditer(text):
        if m.start() > fim:
            brutos.append((text[fim : m.start()], lang))
        lang = None if m.group("close") else _norm_locale(m.group("loc"))
        fim = m.end()

    if fim < len(text):
        brutos.append((text[fim:], lang))

    limpos = []
    for trecho, loc in brutos:
        trecho = ANY_TAG.sub("", trecho)
        if trecho:
            limpos.append((trecho, loc))
    return limpos


def _cut_index(plain, limite):
    """Onde cortar o texto falado: fim de frase se der, senão fim de palavra."""
    if len(plain) <= limite:
        return len(plain), False

    janela = plain[:limite]
    fim = max(janela.rfind("."), janela.rfind("!"), janela.rfind("?"))
    if fim > limite // 2:
        return fim + 1, False

    espaco = janela.rfind(" ")
    return (espaco if espaco > 0 else limite), True


def _render(segs, corte):
    """Monta o SSML com texto escapado e tags sempre balanceadas."""
    partes = []
    usado = 0
    aberto = None

    for trecho, loc in segs:
        if usado >= corte:
            break

        pedaco = trecho[: corte - usado]
        usado += len(pedaco)

        if loc != aberto:
            if aberto:
                partes.append("</lang>")
            if loc:
                partes.append('<lang xml:lang="{}">'.format(loc))
            aberto = loc

        partes.append(escape(pedaco))

    if aberto:
        partes.append("</lang>")

    return "".join(partes)


def build_speech(text):
    """Devolve (ssml para falar, texto puro para o histórico).

    Se o SSML sair inválido por qualquer motivo, cai para texto puro escapado —
    a skill responde errado na pronúncia, mas nunca quebra.
    """
    text = MARKDOWN.sub("", text or "")
    text = re.sub(r"\s+", " ", text).strip()

    segs = _segments(text)
    plain = "".join(t for t, _ in segs)
    corte, cortou_no_meio = _cut_index(plain, MAX_CHARS)
    plain = plain[:corte].strip()

    if cortou_no_meio:
        plain += "..."

    try:
        ssml = _render(segs, corte)
        if cortou_no_meio:
            ssml += "..."
        ElementTree.fromstring("<speak>{}</speak>".format(ssml))  # valida
        return ssml, plain
    except Exception as e:
        logger.warning("SSML invalido, usando texto puro: %s", e)
        return escape(plain), plain


def generate_gpt_response(query):
    try:
        trim_history()
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=MODEL, messages=messages, stream=False, max_tokens=MAX_TOKENS
        )
        ssml, plain = build_speech(response.choices[0].message.content)

        if not plain:
            raise ValueError("resposta vazia do modelo")

        # guarda o texto puro: histórico mais barato e sem tags para confundir
        messages.append({"role": "assistant", "content": plain})

        usage = getattr(response, "usage", None)
        logger.info(
            "deepseek ok | chars=%s | tokens=%s",
            len(plain),
            usage.total_tokens if usage else "?",
        )
        return ssml
    except Exception as e:
        logger.error("deepseek falhou: %s", e, exc_info=True)
        if messages and messages[-1]["role"] == "user":
            messages.pop()  # não guarda pergunta sem resposta
        return "Tive um problema para responder agora. Tenta de novo daqui a pouco."


class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Como posso te ajudar?"

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.CancelIntent")(
            handler_input
        ) or ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Até logo!"

        return handler_input.response_builder.speak(speak_output).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Desculpe, não consegui processar sua solicitação."

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()