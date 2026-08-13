import os
import re
import logging
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

MAX_CHARS = 400  # corte final da fala
MAX_TOKENS = 200  # teto de geração (economiza créditos)
MAX_TURNS = 5  # pares pergunta/resposta mantidos no contexto

SYSTEM_PROMPT = (
    "Você é o assistente pessoal do Rafael e responde por voz na Alexa, "
    "em Português do Brasil. "
    "Fale de forma natural e direta, como numa conversa, e vá direto ao ponto: "
    "primeira frase já responde a pergunta. "
    "Máximo de 400 caracteres, em texto corrido. "
    "Nunca use listas, markdown, emojis, asteriscos, títulos, tabelas ou código. "
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


def clean_for_speech(text):
    """Tira marcações que a Alexa lê mal e corta no fim de frase."""
    text = re.sub(r"[*_`#>|~]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= MAX_CHARS:
        return text

    corte = text[:MAX_CHARS]
    fim = max(corte.rfind("."), corte.rfind("!"), corte.rfind("?"))
    return corte[: fim + 1] if fim > MAX_CHARS // 2 else corte.rsplit(" ", 1)[0] + "..."


def generate_gpt_response(query):
    try:
        trim_history()
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=MODEL, messages=messages, stream=False, max_tokens=MAX_TOKENS
        )
        reply = clean_for_speech(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": reply})

        usage = getattr(response, "usage", None)
        logger.info(
            "deepseek ok | chars=%s | tokens=%s",
            len(reply),
            usage.total_tokens if usage else "?",
        )
        return reply
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