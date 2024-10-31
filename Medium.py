"""
3. Longest Substring Without Repeating Characters
"""
# Apply concept of sliding window and set

from ast import List
from typing import List

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        unique_set = set()
        left = 0
        max_length = 0

        for right in range(len(s)):
            while s[right] in unique_set:
                unique_set.remove(s[left])
                left += 1
            
            unique_set.add(s[right])

            max_length = max(max_length, right - left + 1)

        return max_length

"""
5. Longest Palindromic Substring
"""
class Solution:
    def longestPalindrome(self, s: str) -> str:
        result = ""
        length = 0
        max_length = len(s)

        # Helper function to expand around center
        def expand_around_center(left: int, right: int) -> str:
            while left >= 0 and right < max_length and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        for i in range(max_length):
            # Check for odd-length palindromes
            odd_palindrome = expand_around_center(i, i)
            if len(odd_palindrome) > length:
                result = odd_palindrome
                length = len(odd_palindrome)

            # Check for even-length palindromes
            even_palindrome = expand_around_center(i, i + 1)
            if len(even_palindrome) > length:
                result = even_palindrome
                length = len(even_palindrome)

        return result

"""
6. Zigzag Conversion
"""
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s
        
        rows = [''] * numRows  # Use list comprehension for cleaner initialization
        
        r, move = 0, 1  # `move` will determine direction
        
        for char in s:
            rows[r] += char
            if r == 0:
                move = 1  # Move down
            elif r == numRows - 1:
                move = -1  # Move up
            r += move
        
        # Join all rows into one string
        return ''.join(rows)
"""
7. Reverse Integer
"""  
class Solution:
    def reverse(self, x: int) -> int:
        INT_MAX = 2**31 - 1  # 2147483647
        
        sign = 1 if x > 0 else -1
        x = abs(x)
        result = 0
        while x != 0:
            digit = x % 10
            x //= 10
            if result > (INT_MAX - digit) // 10:
                return 0
            result = result * 10 + digit
        return result * sign

"""
8. String to Integer (atoi)
"""
def myAtoi(s: str) -> int:
    result = 0
    sign = 1
    int_max = 2 ** 31
    isInitial = True

    # Remove leading space
    s = s.lstrip()

    for char in s:
        if char.isnumeric():
            result = result * 10 + int(char)
        elif isInitial:
            if char == '-':
                sign = -1
            elif char == '+':
                sign = 1
            else:
                break
        else:
            break
        isInitial = False

    if sign > 0 and result > int_max - 1:
        result = int_max - 1
    
    if sign < 0 and result > int_max:
        result = int_max
 
    return result * sign

"""
11. Container With Most Water
"""
def maxArea(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        area = min(height[left], height[right]) * (right - left)
        max_area = max(area, max_area)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

"""
12. Integer to Roman
"""
def intToRoman(num: int) -> str:
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    symbols = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    result = ""
    i = 0
    while num > 0:
        # Divide each value on the list
        for _ in range(num // val[i]):
            result += symbols[i]
            num -= val[i]
        # Move along the list
        i += 1
    
    return result

"""
15. 3Sum
"""
def threeSum(nums: List[int]) -> List[List[int]]:
    nums.sort()  # Step 1: Sort the array
    res = []

    # Step 2: Iterate through the array with the fixed element `i`
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:  # Skip duplicates
            continue

        # Initialize two pointers
        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                res.append([nums[i], nums[left], nums[right]])

                # Move the left and right pointers and avoid duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1

            elif total < 0:
                left += 1  # Increase the sum
            else:
                right -= 1  # Decrease the sum

    return res

"""
16. 3Sum Closest
"""
def threeSumClosest(nums: List[int], target: int) -> int:
        nums.sort()  # Sort the array first
        closest_sum = sum(nums[:3])  # Initialize with the sum of the first three elements

        for i in range(len(nums) - 2):
            left, right = i + 1, len(nums) - 1

            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                # Update the closest sum if the current sum is closer to the target
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                # Move pointers to try and get closer to the target
                if current_sum < target:
                    left += 1
                elif current_sum > target:
                    right -= 1
                else:  # If exact target sum is found
                    return target

        return closest_sum

"""
17. Letter Combinations of a Phone Number
"""
from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        
        keyboard = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"]
        }

        result = [""]  # Start with an initial empty combination

        for digit in digits:
            temp = []
            newletters = keyboard.get(digit)
            for item in result:
                for newletter in newletters:
                    temp.append(item + newletter)
            result = temp

        return result

"""
18. 4Sum
"""
