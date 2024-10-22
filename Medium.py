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
