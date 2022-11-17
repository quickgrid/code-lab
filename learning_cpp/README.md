# Learning C++

Notes of my attempts to learn C++ more in depth.

For windows LLVM, MinGW64, ninja, cmake, visual studio, vscode etc. installed or extracted to folder. Also system environment variables path set to path of `*.exe` files. Not all examples require all tools.

Code structure is vscode `Visual Studio` clang format with attempt to follow google style guide best practice where possible. Alternate code formatting if used will be `Chromium`. Header guard follows google style guide `<PROJECT>_<PATH>_<FILE>_H_` full path from project root.

For examples clang version is `15.0.2`, gcc version is `12.2.0` is used and C++20 where possible (`-std=c++20` in compiler).

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

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug "-DCMAKE_C_COMPILER:FILEPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/YOUR_VERSION/bin/Hostx64/x64/cl.exe" "-DCMAKE_CXX_COMPILER:FILEPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/YOUR_VERSION/bin/Hostx64/x64/cl.exe" -SC:\CodeLab\learning_cpp\ex1 -BC:\CodeLab\learning_cpp\ex1\build

Without setting manually.

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug -SC:\CodeLab\learning_cpp\ex1 -BC:\CodeLab\learning_cpp\ex1\build

The build folder will contain `myapp.sln`, `myapp.vcxproj` etc. `myapp.sln` can be opened in visual studio and compile from it. The following command will compile and generate executable from terminal. 

> cmake --build ./build

In `build/Debug` folder `myapp.exe` will be available.

> ./myapp.exe

### Compiling with g++ directly

Compiling `c++` files with `g++` compiler directly and generating executable.

> g++ -O -c myapp.cpp -o myapp

Generate assembly files of various compiler optimization levels for comparison.

> g++ -O3 -S -c myapp.cpp -o myapp_o3.s
> 
> g++ -O -S -c myapp.cpp -o myapp_o.s

Save intermediate files. It will save `*.ii`, `*.o`, `*.s`, `*.exe` files.

> g++ -O3 --save-temps myapp.cpp -o myapp

Define compilation c++ version.

> g++ -O2 -std=c++17  myapp.cpp -o myapp

Disassemble object code with `objdump`. First the object file needs to be generated if not available then disassemble.

> g++ -O3 -c -std=c++17  myapp.cpp

> objdump -d myapp.o

### Using Nvidia NVCC Compiler

It comes with cuda toolkit and as of now should be able to compile `C++17` code.

> nvcc ex1/src/myapp.cpp -o myapp

> ./myapp

# ex2

This project contains with multiple c++ source, header files in various directory. This project directory was structured somewhat following opencv source code. This example uses namespaces to prevent collision with other libraries, includes headers from include directory and manual code testing with assertion etc. 

### Compile C++ Files with Header File in Different Directory with g++

This will include header files from different directories like parent, sibling or others. All c++ files called needs to be passed for compilation. Here, `src\basicmath\include` is path to the header files used.

> g++ -O3 -std=c++17 "-Isrc\basicmath\include" src\basicmath\tests\test_basicmath.cpp src\basicmath\src\adder.cpp -o myapp

# ex3

### Compiling Multiple Header, C++ files in Different Path and Pass by Reference

Example of passing by reference and printing address. It includes multiple header and c++ files in different directories. Though one of the c++ files is not used for executable.

> g++ -O3 -std=c++17 "-Isrc\basicmath\include" "-Isrc\basicmath\src" src\basicmath\tests\test_basicmath.cpp src\basicmath\src\adder.cpp src\basicmath\src\type_utils.cpp -o myapp

# ex4

### Multiple C++ File in Same Namespace and Dynamic Array Allocation, Delete, Vector

Functions defined in both `array_utils.hpp`, `adder.hpp` share same namespace `bmath`. Usage of [fixed width integer types](https://en.cppreference.com/w/cpp/types/integer) in loop and unsigned variable. Dynamic memory array allocation, release and equivalent vector insertion function.

> g++ -O2 -std=c++17 "-Isrc\basicmath\include" src\basicmath\tests\test_array_declaration.cpp -o 
myapp src\basicmath\src\array_utils.cpp

# ex5

Example of using `CMakeLists.txt` to generate multiple executable at once, memory leak, debugging function call stack.

### Generate multiple executable with cmake

Configure and generate project with given generator.

> cmake -DCMAKE_BUILD_TYPE:STRING=Debug "-DCMAKE_C_COMPILER:FILEPATH=PATH_TO\clang.exe" "-DCMAKE_CXX_COMPILER:FILEPATH=PATH_TO\clang++.exe" -S./ex5 -B./ex5/build -G Ninja

Build all executable files in build folder.

> cmake --build ./ex5/build/

### Show Compiler Warning, Force to Fix Warnings Before Compilation in `memory_leak.cpp`

Compile with clang++ with all diagnostics enabled. As seen by output warnings there was no warnings for memory leak.

> clang++ -Weverything ex5/src/memory_leak.cpp -o out

Treat warnings like error. This will not generate executable until errors fixed.

> clang++ -Werror -Weverything ex5/src/memory_leak.cpp -o out

Doing similar with `g++` to see warnings and attempting to compile with `-Werror` flag. Again there was no warnings for memory leak.

> g++ -Wall -Wextra ex5/src/memory_leak.cpp -o out

> g++ -Wall -Wextra -Werror ex5/src/memory_leak.cpp -o out

### Debugging from terminal and VSCode for call stack, variable change in `stack_frame_debug.cpp`

Debugging from terminal with [gdb](https://www.cprogramming.com/gdb.html). [Resource](https://web.mit.edu/gnu/doc/html/gdb_8.html) for call stack and stack frame. Add `-g` flag for gdb debugging. Another faster option is to use vscode debug option.

> g++ -std=c++17 -g -Wall -Wextra ex5/src/stack_frame_debug.cpp -o out

> gdb out

Add break points.

> break 43

> break 34

> break 25

> break 18

> break 10

Run program. If no breakpoints set it execute program and finish.

> run

List 10 lines around the hit break point.

> list

Print value of variable `a`.

> print a

Watch for change in variable `a`.

> watch a

Step in code and show code.

> step
 
> step
 
> print a

> list

Show all the program stack frames in call stack. It will show current function, line number and stack frames for the program in call stack.

> backtrace

Select a frame and move up to outer frame by 1.

> frame 2

> up 1

Print verbose description of current frame.

> info frame

Print local variables of selected frame.

> info locals

Looking into local variable change. This will show the local value of `a`.

> info locals

> up

> info locals

Quit gdb.

> quit

# ex6

Example of using address and undefined [sanitizers](https://clang.llvm.org/docs/UsersManual.html#controlling-code-generation) for runtime memory and undefined error check. Also example of stack overflow, recursion to tail recursion conversion, execution timing with chrono, pass function reference to another function, pass function reference to another with `std::function`, [doxygen](https://www.cs.cmu.edu/~410/doc/doxygen.html) format comments.

### Address and Undefined Sanitizers Usage

These codes `nullptr_dereference.cpp`, `signed_int_overflow.cpp` and `uninitialized_variable_access.cpp` compiles, runs and produces results. Yet running with sanitizer options produces runtime error on mistakes.

In `stack_overflow.cpp` example in debug or with address sanitizer stack overflow error is shown in run time. Code is same as below just name change.

> clang++ -std=c++20 -g -O2 ex6/src/uninitialized_variable_access.cpp -o app.exe

> ./app.exe

> clang++ -std=c++20 -g -O2 ex6/src/uninitialized_variable_access.cpp -o app.exe "-fsanitize=address,undefined" -fno-omit-frame-pointer

> ./app.exe

### Pass Function to Another Function in `tail_recursion_debug.cpp`

Conversion of recursive fibonacci to tail recursive, function timing, passing function with reference and `std::function` and doxygen comments.
