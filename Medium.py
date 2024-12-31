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
    
"""
36. Valid Sudoku
"""
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # Use a set to check for duplicates, ignoring '.'
        def check_duplicate(nums):
            nums = [num for num in nums if num != '.']
            return len(nums) != len(set(nums))

        blocks = [[] for _ in range(9)]
        columns = [[] for _ in range(9)]

        # Iterate over each row and column
        for rowth, row in enumerate(board):
            # Check each row for duplicates
            if check_duplicate(row):
                return False
            
            for i, num in enumerate(row):
                # Update the cell in the corresponding block and column
                if num != '.':
                    blocks[(rowth // 3) * 3 + i // 3].append(num)
                    columns[i].append(num)
        
        # Check each block for duplicates
        for block in blocks:
            if check_duplicate(block):
                return False
        
        # Check each column for duplicates
        for column in columns:
            if check_duplicate(column):
                return False
        
        return True
    
"""
38. Count and Say
"""
class Solution:
    def countAndSay(self, n: int) -> str:

        def convert(RLE: str) -> str:
            result = ""
            count = 1
            
            # Iterate through RLE to count consecutive characters
            for i in range(1, len(RLE)):
                if RLE[i] == RLE[i - 1]:
                    count += 1
                else:
                    # Append the count and character to the result
                    result += str(count) + RLE[i - 1]
                    count = 1
            
            # Append the last group
            result += str(count) + RLE[-1]
            return result

        # Base case
        if n == 1:
            return "1"
        
        # Recursive call
        return convert(self.countAndSay(n - 1))

"""
39. Combination Sum
"""
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def findRemain(remain: int, path: List[int]) -> List[List[int]]:
            # If the remainder is zero, we found a combination
            if remain == 0:
                return [path]
            
            result = []
            for num in candidates:
                # Only proceed if num is less than or equal to the remainder
                if num <= remain:
                    result += findRemain(remain - num, path + [num])
            
            return result

        solutions = findRemain(target, [])
        
        # Remove duplicates
        unique_solutions = set(tuple(sorted(solution)) for solution in solutions)
        
        # Convert back to a list of lists
        return [list(solution) for solution in unique_solutions]

"""
40. Combination Sum II
"""
from typing import List

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()  # Sort to handle duplicates
        results = []

        def dfs(choices: List[int], remain: int, start: int, path: List[int]):
            if remain == 0:
                results.append(path)
                return

            for i in range(start, len(choices)):
                # Skip duplicates
                if i > start and choices[i] == choices[i - 1]:
                    continue

                if choices[i] > remain:
                    break  # No point in continuing if the number is greater than remaining target

                dfs(choices, remain - choices[i], i + 1, path + [choices[i]])

        dfs(candidates, target, 0, [])
        return results

"""
43. Multiply Strings
"""
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        m, n = len(num1), len(num2)
        result = [0] * (m + n)
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                multiple = int(num1[i]) * int(num2[j])
                multiple += result[i + j + 1]
                result[i + j + 1] = multiple % 10
                result[i + j] += multiple // 10
        
        result_str = ''.join(map(str, result)).lstrip('0')
        
        return result_str if result_str else "0"
    
"""
45. Jump Game II
"""
def jump(nums: List[int]) -> int:
    n = len(nums)

    if n <= 1:
        return 0
    
    l = r = 0
    jumps = 0

    while r < n - 1:
        farthest = 0
        for i in range(l, r + 1):
            farthest = max(farthest, nums[i] + i)
        l = r + 1
        r = farthest
        jumps += 1
    
    return jumps

"""
46. Permutations
"""
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def dfs(remains: int, path: List[int]):
            n = len(remains)
            if n == 0:
                result.append(path)
            else:
                for i in range(n):
                    dfs(remains[:i] + remains[i+1:], path + [remains[i]])
        dfs(nums, [])
        return result

"""
47. Permutations II
"""
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()  # Sort to handle duplicates
        result = []
        visited = [False] * len(nums)
        
        def backtrack(path: List[int]):
            # Base case: if the path length equals nums length, we have a full permutation
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            for i in range(len(nums)):
                # Skip used elements or duplicates
                if visited[i] or (i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]):
                    continue
                
                # Mark the current element as used
                visited[i] = True
                # Include nums[i] in the current path
                backtrack(path + [nums[i]])
                # Backtrack and unmark the current element
                visited[i] = False
        
        # Start backtracking with an empty path
        backtrack([])
        return result

"""
48. Rotate Image
"""

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        # By symmetry, only a quater need to start
        for i in range(n // 2):
            for j in range(i, n - 1 - i):
                temp = matrix[i][j]
                # left-bottom to left-up
                matrix[i][j] = matrix[n-1-j][i]
                # right-bottom to left-botom
                matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
                # right-up to right-bottom
                matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
                # left-up to right-up
                matrix[j][n-1-i] = temp
"""
49. Group Anagrams
"""

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    dicts = {}
    
    for word in strs:
        # Sort the word to generate the key
        sorted_word = "".join(sorted(word))
        
        # Group words by their sorted key
        if sorted_word not in dicts:
            dicts[sorted_word] = []
        dicts[sorted_word].append(word)
    
    # Return the grouped dicts as a list of lists
    return list(dicts.values())

"""
50. Pow(x, n)
"""
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            x = 1 / x
            n = -n

        # Divide and conquer
        half = self.myPow(x, n // 2)
        if n % 2 == 0:
            return half * half
        else:
            return half * half * x

"""
53. Maximum Subarray
"""
# Kadane's Algorithm

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        current_sum = max_sum = nums[0]
    
        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        
        return max_sum

"""
54. Spiral Matrix
"""
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        result = []
        top = left = 0
        bottom = len(matrix) - 1
        right = len(matrix[0]) - 1
        
        while top <= bottom and left <= right:

            # Right
            for i in range(left, right + 1):
                result.append(matrix[top][i])
            top += 1

            # Down
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1

            # Left
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1

            # Up
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1

        return result

"""
55. Jump Game
"""
def canJump(nums: List[int]) -> bool:
    farthest = 0
    final = len(nums) - 1
    
    for i in range(final + 1):
        if i > farthest:
            return False
        farthest = max(farthest, nums[i] + i)
    
    return True

"""
56. Merge Intervals
"""
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # Empty input
        if not intervals:  # Handle empty input
            return []

        # Sort
        intervals.sort(key=lambda x: x[0])

        results = [intervals[0]]

        for current in intervals[1:]:
            last = results[-1]

            # Check overlap
            if current[0] <= last[1]:
                # Update interval
                last[1] = max(last[1], current[1])
            else:
                # Append for no overlap
                results.append(current)

        return results

"""
57. Insert Interval
"""
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        results = []
        added = False

        for interval in intervals:

            if added or interval[1] < newInterval[0]:
                results.append(interval)

            # Case 1: newInterval too large
            elif interval[1] < newInterval[0]:
                results.append(interval)
            
            # Case 2: Back end overlaps with newInterval
            elif interval[0] <= newInterval[0] <= interval[1] and interval[1] < newInterval[1]:
                newInterval[0] = interval[0]
            
            # Case 3:
            elif interval[0] <= newInterval[0] and newInterval[1] <= interval[1]:
                results.append(interval)
                added = True
            
            # Case 4: Front end overlaps with newInterval
            elif interval[0] <= newInterval[1] <= interval[1] and newInterval[1] < interval[1]:
                newInterval[1] = interval[1]

            elif newInterval[1] < interval[0]:
                results.append(newInterval)
                results.append(interval)
                added = True
        
        if not intervals or not added:
            results.append(newInterval)

        return results

"""
59. Spiral Matrix II
"""
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        results = [[0 for _ in range(n)] for _ in range(n)]
        top, bottom, left, right = 0, n - 1, 0, n - 1
        content = 1
        
        while content <= n * n:

            # Right
            for i in range(left, right + 1):
                results[top][i] = content
                content += 1
            top += 1

            # Down
            for i in range(top, bottom + 1):
                results[i][right] = content
                content += 1
            right -= 1

            # Left
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    results[bottom][i] = content
                    content += 1
                bottom -= 1

            # Up
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    results[i][left] = content
                    content += 1
                left += 1

        return results

"""
62. Unique Paths
"""

# Possible path combination is a binomial coefficient
# (n + m - 2) C (m - 1)
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        def factorial(k: int) -> int:
            result = 1
            for i in range(2, k + 1):
                result *= i
            return result

        return factorial(m + n - 2) // (factorial(m - 1) * factorial(n - 1))

"""
63. Unique Paths II
"""
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        matrix = [[0 for _ in range(n)] for _ in range(m)]
        matrix[0][0] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    matrix[i][j] = 0
                else:
                    if obstacleGrid[i - 1][j] == 0 and i > 0:
                        matrix[i][j] += matrix[i - 1][j]
                    if obstacleGrid[i][j - 1] == 0 and j > 0:
                        matrix[i][j] += matrix[i][j - 1]

        return matrix[m - 1][n - 1]

"""
64. Minimum Path Sum
"""
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                elif i == 0: # Only add the left
                    grid[i][j] += grid[i][j - 1]
                elif j == 0: # Only add the top
                    grid[i][j] += grid[i - 1][j]
                else: # Compare the min path
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

        return grid[m - 1][n - 1]

"""
71. Simplify Path
"""
class Solution:
    def simplifyPath(self, path: str) -> str:
        parts = path.split("/")
        stack = []

        for part in parts:
            # One level up
            if part == "..":
                if stack:
                    stack.pop()
            # Ignore "."
            elif part and part != ".":
                stack.append(part)
        
        return "/" + "/".join(stack)

"""
72. Edit Distance
"""
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        
        # Swap by size
        if m < n:
            word1, word2 = word2, word1
            m, n = n, m

        # Dynamic programming array
        dp = list(range(n + 1))

        for i in range(1, m + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    # No new operation for identical character
                    dp[j] = prev[j - 1]
                else:
                    # Insert, delete or replace
                    dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
        
        return dp[-1]

"""

"""
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        first_row_zero = any(matrix[0][j] == 0 for j in range(n))
        first_col_zero = any(matrix[i][0] == 0 for i in range(m))
        
        # Use the first row and column as markers
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        # Set matrix elements to zero based on the markers
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        # Handle the first row
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0

        # Handle the first column
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0

"""
74. Search a 2D Matrix
"""
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        min, max = 0, n * m -1

        def convert(index: int) -> int:
            i = index // n
            j = index % n
            return matrix[i][j]
        
        while min <= max:
            mid = (min + max) // 2

            if target == convert(mid) or target == convert(min) or target == convert(max):
                return True
            
            elif target < convert(mid):
                min, max = min + 1, mid - 1
            
            else:
                min, max = mid + 1, max - 1

        return False

"""
75. Sort Colors
"""
def sortColors(nums: List[int]) -> None:
    n = len(nums)
    # Count numbers of each colors
    zero, one = 0, 0

    for num in nums:
        if num == 0:
            zero += 1
        elif num == 1:
            one += 1

    # Overwrite the array
    for i in range(zero):
        nums[i] = 0
    for i in range(zero, zero + one):
        nums[i] = 1
    for i in range(zero + one, n):
        nums[i] = 2

"""
77. Combinations
"""
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []

        # Helper function to manage paths
        def genPath(start: int, k: int, path: List[int]):
            # Stop when length is achieved
            if len(path) == k:
                result.append(path[:])
                return

            # Generate new path starting from start
            for i in range(start, n + 1):
                path.append(i)
                genPath(i + 1, k, path)
                path.pop()
        
        # Initialize
        genPath(1, k, [])

        return result
    
"""
78. Subsets
"""
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = [[]]
        n = len(nums)

        def furtherPath(start: int, path: List[int]):
            if start >= n:
                return
            for i in range(start, n):
                path.append(nums[i])
                result.append(path[:])
                furtherPath(i + 1, path)
                path.pop()
        
        furtherPath(0, [])

        return result

"""
79. Word Search
"""
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])

        def check(i: int, j: int, ptr: int) -> bool:
            if ptr == len(word):  # All characters matched
                return True
            
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[ptr]:
                return False
            
            # Temporarily mark the cell as visited
            temp = board[i][j]
            board[i][j] = '#'

            # Explore all four directions
            found = (
                check(i - 1, j, ptr + 1) or  # Up
                check(i + 1, j, ptr + 1) or  # Down
                check(i, j - 1, ptr + 1) or  # Left
                check(i, j + 1, ptr + 1)     # Right
            )

            # Restore the cell
            board[i][j] = temp

            return found

        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:  # Start the search
                    if check(i, j, 0):
                        return True
        
        return False
    
"""
80. Remove Duplicates from Sorted Array II
"""
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        n = len(nums)
        while i < n:
            # Check with the previous
            if i > 1 and nums[i] == nums[i - 1] and nums[i] == nums[i - 2]:
                del nums[i]
                n -= 1
            else:
                i += 1
        return i

"""
81. Search in Rotated Sorted Array II
"""
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            # Direct hit
            if nums[mid] == target:
                return True

            # Handle duplicates by shrinking bounds
            while left < mid and nums[left] == nums[mid]:
                left += 1

            # Determine which half is sorted
            if nums[left] <= nums[mid]:  # Left half is sorted
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return False

"""
89. Gray Code
"""
def grayCode(n: int) -> List[int]:
    # Set max as 2^n
    total_numbers = 1 << n
    result = []
    
    for i in range(total_numbers):
        gray = i ^ (i >> 1)
        result.append(gray)
    
    return result

"""
90. Subsets II
"""
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        result = [[]]
        n = len(nums)
        nums.sort()

        def furtherPath(start: int, path: List[int]):
            for i in range(start, n):
                # Skip duplicates by checking if the current element is the same as the previous one
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                result.append(path[:])
                furtherPath(i + 1, path)
                path.pop()
        
        furtherPath(0, [])
        return result


class Solution:
    def numDecodings(self, s: str) -> int:
        # Empty string or leading zero
        if not s or s[0] == "0":
            return 0
        
        n = len(s)
        dp = [0] * (n + 1)  # dp[i] means the number of ways to decode s[:i]
        dp[0] = 1  # Base case: empty string has one way to decode
        dp[1] = 1  # Base case: one non-zero digit has one way to decode
        
        for i in range(2, n + 1):
            # Single digit decode
            if s[i - 1] != "0":  # Non-zero digit
                dp[i] += dp[i - 1]
            
            # Two digit decode
            two_digit = int(s[i - 2:i])  # Check the last two digits
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]
        
        return dp[n]

"""
93. Restore IP Addresses
"""
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []
        n = len(s)

        def generate(start: int, layer: int, path: List[str]):
            # Base case: Four segments
            if layer == 4 and start == n:
                result.append('.'.join(path))
                return
            
            # Run out of remain
            if layer == 4 or start == n:
                return
            
            for i in range(1, 4):
                if start + i > n:
                    break

                segment = s[start: start + i]
                
                # Invalid segment
                if (len(segment) > 1 and segment[0] == "0") or int(segment) > 255:
                    continue

                generate(start + i, layer + 1, path + [segment])

        generate(0, 0, [])
        return result

"""
96. Unique Binary Search Trees
"""

class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        # Base case
        dp[0], dp[1] = 1, 1

        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        
        return dp[n]
    
"""
97. Interleaving String
"""
def isInterleave(s1, s2, s3):
    if len(s1) + len(s2) != len(s3):
        return False
    
    # Initialize DP table
    m, n = len(s1), len(s2)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Fill DP table
    for i in range(m + 1):
        for j in range(n + 1):
            if i > 0 and s1[i - 1] == s3[i + j - 1]:
                dp[i][j] |= dp[i - 1][j]
            if j > 0 and s2[j - 1] == s3[i + j - 1]:
                dp[i][j] |= dp[i][j - 1]
    
    return dp[m][n]

