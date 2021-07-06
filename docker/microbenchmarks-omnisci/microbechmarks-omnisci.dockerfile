FROM ubuntu:18.04
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV MODIN_BACKEND "omnisci"
ENV MODIN_EXPERIMENTAL "true"

RUN apt-get update --yes \
 && apt-get install wget --yes \
 && rm -rf /var/lib/apt/lists/*

ENV USER modin
ENV UID 1000
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --home $HOME \
    $USER

ENV CONDA_DIR ${HOME}/miniconda

SHELL ["/bin/bash", "--login", "-c"]

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda3.sh \
 && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
 && "${CONDA_DIR}/bin/conda" init bash \
 && rm -f /tmp/miniconda3.sh \
 && echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

RUN conda update -n base -c defaults conda -y \
 && conda config --set channel_priority strict \
 && conda create -n modin --yes --no-default-packages \
 && conda activate modin \
 && conda install -c conda-forge -c default python==3.8 pip git \
 && pip install git+https://github.com/airspeed-velocity/asv.git@ef016e233cb9a0b19d517135104f49e0a3c380e9 \
 && conda clean --all --yes

RUN git clone https://github.com/modin-project/modin.git

COPY asv_runner.sh "${HOME}/asv_runner.sh"
RUN mkdir /bench_results
RUN echo 'conda activate modin && asv_runner.sh $*' > /entrypoint.sh

ENTRYPOINT ["/bin/bash", "--login", "/entrypoint.sh"]
