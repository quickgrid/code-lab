#include <iostream>
#include <typeinfo>

using namespace std;

void test_1d_array_dynamic_allocation(const int &array_size)
{
    int *arr = new int[array_size];
    for (int k = 0; k < array_size; ++k)
    {
        arr[k] = k * k;
    }

    for (int k = 0; k < array_size; ++k)
    {
        cout << *(arr + k) << "\n";
        cout << (arr + k) << "\n";
        cout << typeid(k).name() << ", " << sizeof(k) << "\n";
        cout << "\n";
    }

    delete[] arr;
    cout << "Done\n";
}

void test_1d_vector(const int &array_size) {}