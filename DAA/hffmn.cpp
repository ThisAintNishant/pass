#include <iostream>
#include <vector>
using namespace std;

// Simple structure to hold item properties
struct Item {
    int weight;
    int value;
};

// Function to solve knapsack problem
int findMaxValue(int capacity, vector<Item>& items) {
    int numItems = items.size();
    
    // Create DP table
    vector<vector<int>> dp(numItems + 1, 
                          vector<int>(capacity + 1, 0));
    
    // Fill DP table
    for (int item = 1; item <= numItems; item++) {
        for (int weight = 0; weight <= capacity; weight++) {
            // Get current item's properties (adjust for 0-based indexing)
            int currentWeight = items[item-1].weight;
            int currentValue = items[item-1].value;
            
            // If current item can fit
            if (currentWeight <= weight) {
                // Choose maximum between:
                // 1. Not taking the item
                // 2. Taking the item + maximum value possible with remaining weight
                dp[item][weight] = max(
                    dp[item-1][weight],
                    dp[item-1][weight - currentWeight] + currentValue
                );
            } else {
                // If item is too heavy, skip it
                dp[item][weight] = dp[item-1][weight];
            }
        }
    }
    
    // Return maximum possible value
    return dp[numItems][capacity];
}

int main() {
    // Get knapsack capacity
    int capacity;
    cout << "Enter knapsack capacity: ";
    cin >> capacity;
    
    // Get number of items
    int numItems;
    cout << "Enter number of items: ";
    cin >> numItems;
    
    // Get items' weights and values
    vector<Item> items(numItems);
    for (int i = 0; i < numItems; i++) {
        cout << "Enter weight and value for item " << i + 1 << ": ";
        cin >> items[i].weight >> items[i].value;
    }
    
    // Calculate and display result
    int result = findMaxValue(capacity, items);
    cout << "Maximum value: " << result << endl;
    
    return 0;
}
/*
    Input ->
    6
    a b c d e f
    5 9 12 13 16 45

    Output ->
    f 0
    c 100
    d 101
    a 1100
    b 1101
    e 111
*/

/*
    Complexity Analysis -->
    Time Complexity -> O(n log(n))
    Space Complexity -> O(n)
*/