FROM atinoda/text-generation-webui:triton
RUN mkdir /app/extensions/PDFOoba
COPY ./requirements.txt /app/extensions/PDFOoba/requirements.txt
RUN pip install -r /app/extensions/PDFOoba/requirements.txt
COPY ./script.py /app/extensions/PDFOoba/script.py
COPY ./summarize.py /app/extensions/PDFOoba/summarize.py
