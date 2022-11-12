# Learning C++

Notes of my attempts to learn C++ more in depth.

For windows LLVM, MinGW64, ninja, cmake, visual studio, vscode etc. installed or extracted to folder. Also system environment variables path set to path of `*.exe` files. Not all examples require all tools.

# ex1

Run these from command line. VSCode editor terminal is a good for editing and testing at the same time.

### Clang Ninja Project

Basic compilation of cpp file with clang llvm in windows with `CMakeLists.txt`. 
Following command will generate ninja project in `build` folder with `build.ninja` in it. Here, `C:\GitHub\CodeLab\learning_cpp\ex1` is the path to `myapp.cpp` file. 
If source path not specified then current path will be used for cpp.

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug "-DCMAKE_C_COMPILER:FILEPATH=C:\Program Files\LLVM\bin\clang.exe" "-DCMAKE_CXX_COMPILER:FILEPATH=C:\Program Files\LLVM\bin\clang++.exe" -SC:\GitHub\CodeLab\learning_cpp\ex1 -BC:\GitHub\CodeLab\learning_cpp\ex1\build -G Ninja

Finally using the following command will generate executable file. Since `myapp` was defined in CMakeLists.txt so it is used.

> ninja myapp

In build folder `myapp.exe` will be generated. Now it can be run with following command.

> ./myapp.exe

### G++ Ninja Project

For compilation with `g++`. Rest of the process to compile and run is same.

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug "-DCMAKE_C_COMPILER:FILEPATH=C:\mingw64\bin\gcc.exe" "-DCMAKE_CXX_COMPILER:FILEPATH=C:\mingw64\bin\g++.exe" -SC:\GitHub\CodeLab\learning_cpp\ex1 -BC:\GitHub\CodeLab\learning_cpp\ex1\build -G Ninja

### Visual Studio Project with Visual C++ Compiler

Setting `cl.exe` path manually. Replace `YOUR_VERSION` with visual c++ compiler version.

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug "-DCMAKE_C_COMPILER:FILEPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/YOUR_VERSION/bin/Hostx64/x64/cl.exe" "-DCMAKE_CXX_COMPILER:FILEPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/YOUR_VERSION/bin/Hostx64/x64/cl.exe" -SC:\Users\computer\Documents\GitHub\CodeLab\learning_cpp\ex1 -BC:\Users\computer\Documents\GitHub\CodeLab\learning_cpp\ex1\build

Without setting manually.

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug -SC:\Users\computer\Documents\GitHub\CodeLab\learning_cpp\ex1 -BC:\Users\computer\Documents\GitHub\CodeLab\learning_cpp\ex1\build

The build folder will contain `myapp.sln`, `myapp.vcxproj` etc. `myapp.sln` can be opened in visual studio and compile from it. The following command will compile and generate executable from terminal. 

> cmake --build ./build

In `build/Debug` folder `myapp.exe` will be available.

> ./myapp.exe
