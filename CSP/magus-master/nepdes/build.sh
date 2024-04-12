c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/nep.cpp -o nepdes$(python3-config --extension-suffix)
