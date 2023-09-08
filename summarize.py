from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.text_generation import get_encoded_length, generate_reply

def summarize_text(text, c_size, c_overlap, final_size, state):
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
    new_text = summaries.join("\n")
    if get_encoded_length(new_text) > get_encoded_length(text):
        return "Error, summary was longer than original. Try a larger final size, larger chunk size, or smaller chunk overlap."
    if get_encoded_length(new_text) > final_size:
        return summarize_text(new_text, c_size, c_overlap, final_size, state)
    return new_text