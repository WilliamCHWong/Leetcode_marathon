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