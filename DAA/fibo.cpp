#include <iostream>
using namespace std;

int fibonacci(int n)
{
    if (n <= 1)
        return n;

    return fibonacci(n - 1) + fibonacci(n - 2);
}

/**
 * Time Complexity: O(2^n)
 * For each call fibonacci(n), it recursively calls fibonacci(n-1) and fibonacci(n-2).
 * This creates a tree of recursive calls where each node represents a function call.
 * The size of this recursive call tree is exponential, and the time complexity is O(2^n)
 * because for each n, you essentially have two branches (n-1 and n-2).

 * Space Complexity: O(n)
 * The recursion depth is proportional to n because it will reach the base case after n
 * recursive calls. Since there is no extra memory allocated except for the recursive call
 * stack, the space complexity is linear.
 */
void recursive(int n)
{
    for (int i = 0; i < n; ++i)
        cout << fibonacci(i) << " ";
}

/**
 * Time Complexity: O(n)
 * Space Complexity: O(1)
 */
void iterative(int n)
{
    int a1 = 0;
    int a2 = 1;

    cout << a1 << " " << a2 << " ";

    for (int i = 2; i < n; i++)
    {
        int a3 = a2 + a1;
        a1 = a2;
        a2 = a3;
        cout << a2 << " ";
    }

    cout << endl;
}

int main()
{
    int n;
    cout<<"Enter the number of terms"<<endl;
    cin>>n;
    recursive(n);
    cout << endl;
    iterative(n);
    return 0;
}