# To build fast independent cascade
```
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -I../extern/pybind11/include -I/usr/include/python3.12  independent_cascade.cpp -o fast_ic$(python3-config --extension-suffix)
```
Then you have to append the folder containing the .so file to the PYTHONPATH and PATH
```
export PYTHONPATH=$PYTHONPATH:/home/yic033/ECE227/src
export PATH=$PATH:/home/yic033/ECE227/src
```