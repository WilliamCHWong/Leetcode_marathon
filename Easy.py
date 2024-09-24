# Two Sum
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        record = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in record:
                return [record[complement], i]
            record[num] = i
        return []
    
# Palindrome Number
# Given an integer x, return true if x is a palindrome, and false otherwise.

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        if x < 10:
            return True
        copy = x
        mirror = 0
        while copy > 0:
            mirror = mirror * 10 + (copy % 10)
            copy //= 10
        return x == mirror

"""
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
"""
class Solution:
    def romanToInt(self, s: str) -> int:
        result = 0
        for i in range(len(s)):
            if s[i] == 'I':
                result += 1
            elif s[i] == 'V':
                result += 5
                if s[ i - 1] == 'I' and i > 0:
                    result -= 2
            elif s[i] == 'X':
                result += 10
                if s[ i - 1] == 'I' and i > 0:
                    result -= 2
            elif s[i] == 'L':
                result += 50
                if s[ i - 1] == 'X' and i > 0:
                    result -= 20
            elif s[i] == 'C':
                result += 100
                if s[ i - 1] == 'X' and i > 0:
                    result -= 20
            elif s[i] == 'D':
                result += 500
                if s[ i - 1] == 'C' and i > 0:
                    result -= 200
            elif s[i] == 'M':
                result += 1000
                if s[ i - 1] == 'C' and i > 0:
                    result -= 200
            else:
                result += 0
        return result

"""
Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".
"""
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        result = ""
        for i in range(len(strs[0])):
            for j in range(1, len(strs)):
                if i >= len(strs[j]) or strs[j][i] != strs[0][i]:
                    return result
            result += strs[0][i]
        return result
    
"""
Merge Two Sorted Lists
"""
from typing import Optional
# Definition for singly-linked list.
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy

        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        if list1:
            current.next = list1
        elif list2:
            current.next = list2

        return dummy.next

"""
Valid Parentheses
"""
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        checkList = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in checkList:
                if stack and stack[-1] == checkList[char]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(char)
                
        return len(stack) == 0

class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0 or x == 1:
            return x

        low = 0
        high = x

        while high - low > 1:
            mid = (low + high) // 2
            if mid * mid > x:
                high = mid
            else:
                low = mid
        
        return low