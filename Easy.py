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

        # Merge sort algorithm
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
        # Dictionay makes pairing more efficient
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

"""
Find the nearest integer of the square root of a non-zero integer
"""

class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0 or x == 1:
            return x

        low = 0
        high = x

        # Binary search to reduce complexity
        while high - low > 1:
            mid = (low + high) // 2
            if mid * mid > x:
                high = mid
            else:
                low = mid
        
        return low
    
"""
Remove Duplicates from Sorted Array
pop() has complexity of n, making the solution n^2
use two pointers
"""

class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        if not nums:
            return 0
        unique_pos = 0

        for i in range(1, len(nums)):
            if nums[i] != nums[unique_pos]:
                unique_pos += 1
                nums[unique_pos] = nums[i]

        nums[:] = nums[:unique_pos + 1]
        return unique_pos + 1

"""
Remove Element
use two pointers
"""

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if not nums:
            return 0
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        nums[:] = nums[:k]
        return k
    
"""
Find the Index of the First Occurrence in a String



"""
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        hayLen = len(haystack)
        neeLen = len(needle)
        
        if hayLen == 0 or neeLen == 0 or neeLen > hayLen:
            return -1
        
        for i in range (0, hayLen):
            if haystack[i] == needle[0] and (hayLen - i) >= neeLen:
                result = i
                isFull = True
                for j in range(0, len(needle)):
                    if needle[j] != haystack[i + j]:
                        isFull = False
                if isFull:
                    return result
        return -1
    
"""
Search Insert Position
"""

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1

        if target <= nums [0]:
            return 0
        elif target == nums[high]:
            return high
        elif target > nums[high]:
            return high + 1
        
        while high > low + 1:
            mid = (high + low) //2
            if target > nums[mid]:
                low = mid
            elif target < nums[mid]:
                high = mid
            else:
                return mid
        return high

"""
Length of last word
"""
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        n = len(s)
        i = n - 1
        # Skip trailing spaces
        while i >= 0 and s[i] == " ":
            i -= 1

        # Count the length of the last word
        length = 0
        while i >= 0 and s[i] != " ":
            length += 1
            i -= 1
            
        return length

"""
Plus One
"""
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits) - 1 
        i = n
        while i >= 0:
            if digits[i] != 9:
                digits[i] += 1
                return digits
            else:
                digits[i] = 0
                i -= 1
        digits.append(0)
        n += 1
        while n > 0:
            digits[n] = digits[n - 1]
            n -= 1
        digits[0] = 1
        return 

"""
Add Binary
"""
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        # Convert to decimal numbers
        sum = int(a, 2) + int(b, 2)
        # Convert to string and remove prefix 0b
        return bin(sum)[2:]

"""
Climbing Stairs
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0:
            return 0
        elif n == 1:
            return 1
        
        def fibonacci(n):
            phi = (1 + 5**0.5) / 2
            psi = (1 - 5**0.5) / 2
            return int((phi**n - psi**n) / (5**0.5))
        
        return fibonacci(n + 1)
    
"""
Remove duplicates from sorted list
"""

# Definition for singly-linked list.

class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:  # Check if the list is empty
            return None
        
        current = head
        while current and current.next:
            if current.next.val == current.val:
                current.next = current.next.next  # Skip the duplicate
            else:
                current = current.next  # Move to the next distinct element
        return head

"""
Merge Sorted Array
"""
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i = m + n - 1
        j = m - 1
        k = n - 1
        while k >= 0:
            if j >= 0 and nums1[j] >= nums2[k]:
                nums1[i] = nums1[j]
                nums1[j] = 0
                j -= 1
            else:
                nums1[i] = nums2[k]
                k -= 1
            i -= 1

"""
Binary Tree Inorder Traversal
"""
# Definition for a binary tree node.
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

"""
Same Tree
"""
# Definition for a binary tree node.
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # If both nodes are None, trees are the same
        if not p and not q:
            return True
        # If one of the nodes is None, trees are not the same
        if not p or not q:
            return False
        # Check if current node values are the same and recurse on children
        return (p.val == q.val) and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
