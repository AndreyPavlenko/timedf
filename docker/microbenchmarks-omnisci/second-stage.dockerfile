FROM amyachev-test:latest

# include also workaround for incompatible ray
RUN conda activate modin_on_omnisci \
 && pip install git+https://github.com/airspeed-velocity/asv.git@ef016e233cb9a0b19d517135104f49e0a3c380e9 \
 && conda uninstall ray-core -c conda-forge \
 && pip install ray==1.4.0

# TODO: add more automatization
# OS=`cat /etc/os-release | grep PRETTY_NAME | grep -o '\".*\"'`
RUN cd modin/asv_bench \
 && conda activate modin_on_omnisci \
 && asv machine --machine docker-omnisci \
    --os "Ubuntu 20.04.2 LTS" \
    --arch "x86_64" \
    --cpu 112 \
    --ram 1000GB

RUN cd modin/asv_bench \
 && conda activate modin_on_omnisci \
 && asv machine --machine docker-python \
    --os "Ubuntu 20.04.2 LTS" \
    --arch "x86_64" \
    --cpu 112 \
    --ram 1000GB
