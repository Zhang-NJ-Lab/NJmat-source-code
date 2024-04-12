# NEP Descriptor
get NEP Descriptor

## Install
### By cmake
Put your path to pybind11 in `add_subdirectory($PYBIND_PATH)`
```shell
mkdir build
cd build
cmake ..
make
```
### By make
Need pybind11, which can be installed by pip directly

```shell
pip install pybind11
```

```shell
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/nep.cpp -o nepdes$(python3-config --extension-suffix)
```
