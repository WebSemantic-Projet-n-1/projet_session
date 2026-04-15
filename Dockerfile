FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

COPY . /workspace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
