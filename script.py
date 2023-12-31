import gradio as gr
import fitz
import re
from html import escape

from modules.ui import gather_interface_values, list_interface_input_elements
from modules.utils import gradio
from modules.text_generation import get_encoded_length
from modules import shared
from extensions.PDFOoba.summarize import summarize_text, summarize_text_to_size

params = {
    "display_name" : "PDFOoba",
    "is_tab" : True
}

def preprocess(text):
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(fileObject):
    if fileObject is None:
        return "No pdf is uploaded", 0
    doc = fitz.open(fileObject.name)
    text = "\n\n".join([preprocess(doc.load_page(i).get_text("text")) for i in range(doc.page_count)])
    doc.close()
    if shared.tokenizer is None:
        return "No model/tokenizer is loaded", 0
    else:
        return text, get_encoded_length(text)

def format_text_html(text):
    enc_length = "No model/tokenizer is loaded" if shared.tokenizer is None else get_encoded_length(text)
    formatted_text = "<br>".join(escape(text).splitlines())
    return f"<br>Tokens: {escape(str(enc_length))}<br><br>Summarized Text:<br>{formatted_text}"

def generate_summary(*args):
    for out in summarize_text(*args):
        yield format_text_html(out)

def generate_summary_sized(*args):
    for out in summarize_text_to_size(*args):
        yield format_text_html(out)

def ui():
    state = gr.State({})
    with gr.Row():
        with gr.Column():
            f = gr.File(
                label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf']
            )
            import_button = gr.Button("Import text from PDF")
            with gr.Tab(label="Summarize"):
                chunk_size_slider = gr.Slider(512, 4096, value=1024, step=64, label="Chunk Size (Tokens)")
                chunk_overlap_slider = gr.Slider(0, 256, value=0, label="Chunk Overlap (Tokens)")
                end_size_slider = gr.Slider(128, 65536, value=4096, step=64, label="Desired Text Size (Tokens)")
                summarize_once_button = gr.Button("Summarize once", variant="primary")
                summarize_until_desired_button = gr.Button("Summarize until Desired Text Size", variant="primary")
                cancel_button = gr.Button("Cancel")
            with gr.Tab(label="Query"):
                pass
            loaded_text = gr.Textbox(label="Loaded text:", max_lines=20, interactive = False)
            token_count_textbox = gr.Textbox(label="Token count:", interactive = False)
        with gr.Column():
            output = gr.HTML(label="Output", show_label=True, value="")

        summarize_event = summarize_once_button.click(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            generate_summary,
            inputs = [loaded_text, chunk_size_slider, chunk_overlap_slider, state],
            outputs = [output]
        )

        summarize_until_desired_event = summarize_until_desired_button.click(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            generate_summary_sized,
            inputs = [loaded_text, chunk_size_slider, chunk_overlap_slider, end_size_slider, state],
            outputs = [output]
        )

        import_event = import_button.click(
            pdf_to_text,
            inputs = [f],
            outputs = [loaded_text, token_count_textbox],
            cancels=[summarize_event, summarize_until_desired_event]
        )

        def cancel_summaries():
            pass

        cancel_button.click(
            cancel_summaries,
            inputs=None,
            outputs=None,
            cancels=[summarize_event, summarize_until_desired_event, import_event]
        )
        
        