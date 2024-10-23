"""
3. Longest Substring Without Repeating Characters
"""
# Apply concept of sliding window and set

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

