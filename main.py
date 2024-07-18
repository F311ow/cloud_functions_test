import functions_framework
from flask import abort
import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory
)


@functions_framework.http
def chat_llm(request):
    generation_config = GenerationConfig(max_output_tokens=300,
                                         temperature=1,
                                         top_p=0.95)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    }
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_args and request.args.get('input_text').strip():
        input_text = request.args.get('input_text')
        history = request.args.get('history')
        format = request.args.get('format')
    elif request_json and request_json['input_text'].strip():
        input_text = request_json['input_text']
        history = request_json['history']
        format = request_json['format']
    else:
        return abort(400)
    format = 'text' if format is None else format
    history = [] if history is None else history
    chat_history = []
    for record in history:
        chat_history.append(Content.from_dict(record))
    vertexai.init(project='steady-computer-429115-i6', location='us-central1')
    model = GenerativeModel("gemini-1.0-pro")
    chat = model.start_chat(history=chat_history)
    response = chat.send_message(input_text,
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)
    if format == 'text':
        return response.text
    elif format == 'content':
        return response.candidates[0].content.to_dict()
