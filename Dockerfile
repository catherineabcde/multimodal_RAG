FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
# FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel



RUN apt-get update && apt-get install -y git wget curl debianutils

#RUN apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev pkg-config
#ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"
# $(dpkg -L tesseract-ocr-eng | grep tessdata$)
#RUN echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"

RUN groupadd -g 1000 customgroup 
RUN useradd -m -u 1000 -g customgroup customuser

WORKDIR /work
COPY ./requirements.txt /work
RUN chown -R customuser:customgroup /work

USER customuser

# Change Huggingface download folder
ENV HF_HOME="/workspaces/multimodal-RAG/hf_home"

ENV PATH="/home/customuser/.local/bin:${PATH}"
RUN pip install -r requirements.txt

#USER root

CMD [ "/bin/bash" ]
