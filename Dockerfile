FROM crestedibis/python3-opencv3-docker

RUN apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic \
    mecab-ipadic-utf8

RUN pip install \
        pandas \
        tqdm \
        pyyaml \
        sklearn \
        scipy \
        xmltodict \
        mecab-python3 \
        gensim

COPY . /usr/workspace
WORKDIR /usr/workspace

CMD ["bash"]







