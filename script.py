import gradio as gr
import fitz
import re

from modules.ui import gather_interface_values, list_interface_input_elements
from modules.utils import gradio
from extensions.PDFOoba.summarize import summarize_text

params = {
    "display_name" : "PDFSummaryOoba",
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
    return text

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
                end_size_slider = gr.Slider(128, 4096, value=1024, step=64, label="Desired Text Size (Tokens)")
                summarize_button = gr.Button("Summarize", variant="primary")
            with gr.Tab(label="Query"):
                pass
        with gr.Group():
            output = gr.Textbox(label='Output:')

        summarize_button.click(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            summarize_text,
            inputs=[uploadedText, chunk_size_slider, chunk_overlap_slider, end_size_slider, state],
            outputs=[output]
        )
        
        f.upload(
            pdf_to_text,
            inputs = [f],
            outputs = [uploadedText]
        )