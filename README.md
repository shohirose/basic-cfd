# basic-cfd: C++ source codes for basic CFD
C++ source codes for basic computational fluid dynamics.
Problems are taken from [1].

1-D scalar advection equation with constant velocity

$$
\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0
$$

is solved by using the finie difference method. Currently, the following discretization schemes are implemented:

- FTCS
- Lax scheme
- Lax-Wendroff scheme
- 1st order upwind scheme
- 2nd order upwind scheme
- Harten-Yee non-MUSCL TVD scheme (2nd order)
- MUSCL TVD scheme with minmod limiter (3rd order)

Please refer to [1] for the details of each scheme.

# How to compile

Run the following commands under the root directory to build the project.

```
$ cmake -S . -B build
$ cmake --build build
```

Please note that the project depends on Eigen and fmt libraries, which are automatically downloaded and built by CMake using the `FetchContent` module.

Then, run the following commands to create figures under `results` directory.

```
$ ./build/advection_1d
$ python plot.py
```

# References
1. 藤井孝藏 立川智章 (2020)「Pythonで学ぶ流体力学の数値計算法」
