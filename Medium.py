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
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result, quadruplet = [], []

        def Sum(k, start, target):
            # Base case (Two sum problem)
            if k == 2:
                left, right = start, len(nums) - 1
                # Two pointers
                while left < right:
                    current_sum = nums[left] + nums[right]
                    if current_sum < target:
                        left += 1
                    elif current_sum > target:
                        right -= 1
                    else:
                        result.append(quadruplet + [nums[left], nums[right]])
                        left += 1
                        right -= 1
                        # Skip duplicates
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
            else:
                for i in range(start, len(nums) - k + 1):
                    # Skip duplicates
                    if i > start and nums[i] == nums[i - 1]:
                        continue
                    quadruplet.append(nums[i])
                    # Recursive call to reduce k
                    Sum(k - 1, i + 1, target - nums[i])
                    quadruplet.pop()
        
        Sum(4, 0, target)
        
        # Remove duplicate quadruplets
        unique_result = []
        for quad in result:
            if quad not in unique_result:
                unique_result.append(quad)
        
        return unique_result

"""
22. Generate Parentheses
"""
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        
        # Helper function for backtracking
        def backtrack(current_string, open_count, close_count):
            # Base case: used all open and close
            if len(current_string) == 2 * n:
                result.append(current_string)
                return
            
            # Add open when legitimate
            if open_count < n:
                backtrack(current_string + "(", open_count + 1, close_count)
            
            # Add close when legitimate
            if close_count < open_count:
                backtrack(current_string + ")", open_count, close_count + 1)
        
        backtrack("", 0, 0)
        return result

"""
29. Divide Two Integers
"""
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:

        # Handle division by zero
        if divisor == 0:
            raise ValueError("Division by zero is not allowed.")
        
        # Constants for 32-bit signed integer limits
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        
        # Handle overflow case where result would exceed 32-bit integer limit
        if dividend == INT_MIN and divisor == -1:
            return INT_MAX

        # Determine the sign of the result
        sign = 1 if (dividend > 0) == (divisor > 0) else -1
        
        # Work with absolute values to simplify division logic
        dividend, divisor = abs(dividend), abs(divisor)
        result = 0
        
        # Subtract divisor from dividend repeatedly using bit shifting
        while dividend >= divisor:
            temp, multiple = divisor, 1
            while dividend >= (temp << 1):
                temp <<= 1
                multiple <<= 1
            dividend -= temp
            result += multiple

        # Apply sign to result
        result = result if sign > 0 else -result
        
        # Clamp the result within 32-bit signed integer bounds
        return min(max(result, INT_MIN), INT_MAX)

"""
31. Next Permutation
"""

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n - 2
        
        # Step 1: Find the first decreasing element from the end
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:  # If there is a decreasing element
            # Step 2: Find the smallest element greater than nums[i] to the right of it
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            # Swap the two elements
            nums[i], nums[j] = nums[j], nums[i]
        
        # Step 3: Reverse the subarray to the right of index i
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

"""
33. Search in Rotated Sorted Array
"""
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Check if the middle element is the target
            if nums[mid] == target:
                return mid
            
            # Determine if the left half is sorted
            if nums[left] <= nums[mid]:
                # Check if the target is in the left half
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # Otherwise, the right half must be sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        # Target not found
        return -1

"""
34. Find First and Last Position of Element in Sorted Array
"""
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        max = len(nums) - 1
        left, right = 0, max
      
        while left <= right:
            mid = (left + right) // 2
            
            # Check if the middle element is the target
            if nums[mid] == target:
                first, last = mid, mid
                while nums[first] == target and first >= 0:
                    first -= 1
                while last <= max and nums[last] == target:
                    last += 1
                return [first + 1, last - 1]
            
            # Determine if the left half is sorted
            if target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        
        # Target not found
        return [-1, -1] 