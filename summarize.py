from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.text_generation import get_encoded_length, generate_reply

def summarize_text(text, c_size, c_overlap, state):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = c_size,
        chunk_overlap  = c_overlap,
        length_function = get_encoded_length
    )
    chunks = text_splitter.create_documents([text])
    summaries = []
    for chunk in chunks:
        prompt = f"Text:\n\n{chunk}\n\nInstructions: Summarize the text above. Use as few words as possible while keeping all the information in the text. Respond with your summary and nothing else.\n\nSummary:"
        for a in generate_reply(prompt, state):
            if isinstance(a, str):
                answer = a
            else:
                answer = a[0]
        summaries.append(answer)
    new_text = "\n".join(summaries)
    return new_text, get_encoded_length(new_text)

def summarize_text_to_size(text, c_size, c_overlap, final_size, state):
    if text is None or text == "":
        yield ""
    else:
        new_text, new_encode_len = summarize_text(text, c_size, c_overlap, state)
        yield new_text, new_encode_len
        if new_encode_len > get_encoded_length(text):
            yield "Error, summary was longer than original. Try a larger desired size, larger chunk size, or smaller chunk overlap."
        else if new_encode_len > final_size:
            yield from summarize_text_to_size(new_text, c_size, c_overlap, final_size, state)
