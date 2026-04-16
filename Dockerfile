FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN apt-get update && apt-get install -y git unzip && apt-get clean && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

COPY . /workspace

EXPOSE 8891
