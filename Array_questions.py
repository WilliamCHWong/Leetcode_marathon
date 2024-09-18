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
