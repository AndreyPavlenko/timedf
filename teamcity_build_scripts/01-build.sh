. /usr/local/mapd-deps/mapd-deps.sh

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release -DENABLE_CUDA=off -DENABLE_AWS_S3=off ..
make -j 8

cp /localdisk/benchmark_datasets/omnisci.conf .
mkdir data
./bin/initdb --data data

mkdir ibis
cd ibis
git clone https://github.com/ibis-project/ibis.git ./ibis
pip install -r requirements.txt
python3 setup.py build
cd ..
