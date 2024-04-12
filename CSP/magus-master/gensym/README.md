# gensym
Generate structures with specific spacegroups and constraints.

## Install
### By cmake
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

Note:
[0]use pybind instead of boost since commit 9d56c269.
[1]runtime log is removed since commit 194bc56. If you want debug infos, use -DDEBUG.


```shell
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) test.cpp -o py2cpp$(python3-config --extension-suffix)
```
