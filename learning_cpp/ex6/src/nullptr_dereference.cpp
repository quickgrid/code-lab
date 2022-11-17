// https://hackingcpp.com/cpp/tools/asan.html
#include <iostream>

int main()
{
    int *p = nullptr;
    std::cout << p << " " << *p << "\n";
}
