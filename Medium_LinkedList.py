# Definition for singly-linked list.
from typing import Optional
from typing import List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        last_digit = ListNode()
        current = last_digit
        carry = 0
        while l1 or l2:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0

            v3 = v1 + v2 + carry
            carry = v3 // 10
            v3 %= 10
            current.next = ListNode(v3)

            current = current.next
            l1 = l1.next if l1 else None
            l2 = l2.next  if l2 else None
        if carry > 0:
            current.next = ListNode(carry)
        return last_digit.next
    
"""
19. Remove Nth Node From End of List
"""
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # Handle edge case: remove head
        dummy = ListNode(0)
        dummy.next = head
        fast, slow = dummy, dummy

        # Move fast pointer n + 1 steps ahead
        for _ in range(n + 1):
            fast = fast.next
        
        # Move both pointers till the end
        while fast:
            fast = fast.next
            slow = slow.next
        
        # Remove
        slow.next = slow.next.next
        
        return dummy.next

