from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.text_generation import get_encoded_length, generate_reply
from modules import shared

def summarize_text(text, c_size, c_overlap, state):
    if shared.model is None or shared.tokenizer is None:
        yield "Model / tokenizer not loaded"
    elif text is None or text == "":
        yield "No text to summarize"
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = c_size,
            chunk_overlap  = c_overlap,
            length_function = get_encoded_length
        )
        chunks = text_splitter.create_documents([text])
        summaries = []
        for i in range(len(chunks)):
            prompt = f"Text:\n\n{chunks[i]}\n\nInstructions: Summarize the text above. Respond with your summary and nothing else.\n\nSummary:"
            for a in generate_reply(prompt, state):
                if isinstance(a, str):
                    answer = a
                else:
                    answer = a[0]
            summaries.append(answer)
            yield "\n".join(summaries+chunks[i+1:])

def summarize_text_to_size(text, c_size, c_overlap, final_size, state):
    if shared.model is None or shared.tokenizer is None:
        yield "Model / tokenizer not loaded"
    elif text is None or text == "":
        yield "No text to summarize"
    else:
        encode_len = final_size+1 # Always guarantee one summary
        while encode_len > final_size:
            for s in summarize_text(text, c_size, c_overlap, state):
                yield s
                summary = s
            encode_len = get_encoded_length(summary)