"""
141. Linked List Cycle
"""

# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# Linear
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        record = []
        current = head
        while current.next :
            if current.next in record:
                return True
            else:
                record.append(current.next)
                current = current.next
        return False

# Tortoise and Hare algorithm
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = slow = head  # Combine initialization
        
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next  # Simultaneous assignment
            
            if slow == fast:
                return True
        
        return False