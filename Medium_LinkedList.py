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
"""
24. Swap Nodes in Pairs
"""
def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    while prev.next and prev.next.next:
        # Nodes to be swapped
        first = prev.next
        second = first.next
        
        # Swapping
        first.next = second.next
        second.next = first
        prev.next = second
        
        # Move `prev` two nodes ahead
        prev = first
    
    return dummy.next

"""
61. Rotate List
"""
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head

        # Calculate the length of the list
        length = 1
        current = head
        while current.next:
            current = current.next
            length += 1

        # Normalize k
        k %= length
        if k == 0:
            return head

        # Find the new head
        steps_to_new_head = length - k
        new_tail = head
        for _ in range(steps_to_new_head - 1):
            new_tail = new_tail.next

        # Rotation
        new_head = new_tail.next
        new_tail.next = None
        current.next = head

        return new_head

