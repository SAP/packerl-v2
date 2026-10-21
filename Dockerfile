FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y \
        wget \
        git \
        gcc \
        g++ \
        cmake \
        bzip2 \
        python3 \
        libboost-all-dev \
    	libprotobuf-dev \
	protobuf-compiler \
    	pybind11-dev \ 
    	python3-pip 

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
	bash Miniforge3.sh -b -p "${HOME}/conda" 

WORKDIR /packerl 

COPY ./ /packerl

RUN source "${HOME}/conda/etc/profile.d/conda.sh" && \
	conda env create -f environment.yaml && \
	conda activate packerl 

RUN source "${HOME}/conda/etc/profile.d/conda.sh" && \
	conda activate packerl && \
	bash ./install.sh -p cpu

RUN apt-get clean && \
    	rm -rf /var/lib/apt 


#ENTRYPOINT ["/bin/bash", "-c", "source /root/conda/etc/profile.d/conda.sh"]
#CMD ["/bin/bash", "/packerl/entrypoint.sh"]
#CMD ["/bin/bash", "/packerl/entrypoint.sh"]
#CMD ["python", "main.py", "+_runs=sanity_check_single"]


