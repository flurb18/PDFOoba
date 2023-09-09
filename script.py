import gradio as gr
import fitz
import re

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
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(fileObject):
    doc = fitz.open(fileObject.name)
    text = "\n\n".join([preprocess(doc.load_page(i).get_text("text")) for i in range(doc.page_count)])
    doc.close()
    if shared.tokenizer is None:
        enc_length = "No model/tokenizer is loaded"
    else:
        enc_length = get_encoded_length(text)
    return text, enc_length

def ui():
    state = gr.State({})
    uploadedText = gr.State("")
    with gr.Row():
        with gr.Column():
            f = gr.File(
                label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf']
            )
            with gr.Tab(label="Summarize"):
                chunk_size_slider = gr.Slider(256, 2048, value=256, step=64, label="Chunk Size (Tokens)")
                chunk_overlap_slider = gr.Slider(0, 128, value=0, label="Chunk Overlap (Tokens)")
                end_size_slider = gr.Slider(128, 65536, value=1024, step=64, label="Desired Text Size (Tokens)")
                summarize_once_button = gr.Button("Summarize once", variant="primary")
                summarize_until_desired_button = gr.Button("Summarize until Desired Text Size", variant="primary")
                cancel_button = gr.Button("Cancel")
            with gr.Tab(label="Query"):
                pass
        with gr.Group():
            token_count_textbox = gr.Textbox(label="Token count:", interactive = False)
            output = gr.Textbox(label='Output:', interactive = False)

        summarize_event = summarize_once_button.click(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            summarize_text,
            inputs = [output, chunk_size_slider, chunk_overlap_slider, state],
            outputs = [output, token_count_textbox]
        )

        summarize_until_desired_event = summarize_until_desired_button.click(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            summarize_text_to_size,
            inputs = [output, chunk_size_slider, chunk_overlap_slider, end_size_slider, state],
            outputs = [output, token_count_textbox]
        )

        upload_event = f.upload(
            pdf_to_text,
            inputs = [f],
            outputs = [output, token_count_textbox],
            cancels=[summarize_event, summarize_until_desired_event]
        )

        def cancel_summaries():
            pass

        cancel_button.click(
            cancel_summaries,
            inputs=None,
            outputs=None,
            cancels=[summarize_event, summarize_until_desired_event, upload_event]
        )
        
        