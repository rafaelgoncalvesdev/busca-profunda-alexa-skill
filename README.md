# Busca Profunda - Alexa Skill

Integração Alexa + Deepseek API. Faça perguntas via voz e receba respostas com IA em português.

## Quick Start

1. Clone este repositório
2. Configure sua API key Deepseek em `lambda/lambda_function.py` (linha 17)
3. Deploy: `ask deploy`

## Stack

- **Runtime**: Python 3.x (AWS Lambda)
- **SDK**: Alexa Skills Kit (ask-sdk-core)
- **API**: Deepseek Chat
- **Locales**: Português (Brasil)

## Arquitetura

```
Alexa Device → AWS Lambda → Deepseek API → Resposta em Voz
```

## Comandos

```powershell
# Deploy completo
ask deploy

# Deploy apenas Lambda
ask deploy --target lambda

# Testar localmente
ask dialog --locale pt-BR
```

## Configuração

- **Invocation**: "chat avançado"
- **Intent Principal**: GptQueryIntent
- **Limite Resposta**: 400 caracteres (configurável)

## Licença

MIT
