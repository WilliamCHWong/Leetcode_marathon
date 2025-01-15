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

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        
        # Make dummy to handle head removal
        dummy = ListNode(0, head)
        slow = dummy
        fast = head

        while fast:
            # Check duplicate
            if fast.next and fast.val == fast.next.val:
                # Skip all duplicates
                while fast.next and fast.val == fast.next.val:
                    fast = fast.next
                # Removal action
                slow.next = fast.next
            else:
                slow = slow.next
            fast = fast.next

        return dummy.next

"""
86. Partition List
"""
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        # Dummy nodes for the two partitions
        less_head = ListNode(0)
        greater_head = ListNode(0)
        
        # Pointers to build the two lists
        less = less_head
        greater = greater_head
        
        # Traverse the original list
        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                greater.next = head
                greater = greater.next
            head = head.next
        
        # Ensure the last node of the greater list points to None
        greater.next = None
        
        # Link the two partitions
        less.next = greater_head.next
        
        return less_head.next

"""
92. Reverse Linked List II
"""
def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    prev = dummy

    for _ in range(left - 1):
        prev = prev.next

    reverse_start = prev.next
    current = reverse_start.next

    for _ in range(right - left):
        reverse_start.next = current.next
        current.next = prev.next
        prev.next = current
        current = reverse_start.next
    
    return dummy.next

"""
95. Unique Binary Search Trees II
"""
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        # Base case
        if n == 0:
            return []
        
        def helpGenerate(start, end):
            # Base Case
            if start > end:
                return [None]
            
            result = []
            
            for i in range(start, end + 1):
                left_trees = helpGenerate(start, i - 1)
                right_trees = helpGenerate(i + 1, end)

                for left in left_trees:
                    for right in right_trees:
                        root = TreeNode(i)
                        root.left = left
                        root.right = right
                        result.append(root)
            
            return result
        
        return helpGenerate(1, n)

"""
98. Validate Binary Search Tree
"""
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def checkNodes(node: TreeNode, low: float, high: float) -> bool:
            if not node:
                return True
            
            if not (low < node.val < high):
                return False
            
            return checkNodes(node.left, low, node.val) and checkNodes(node.right, node.val, high)

        return checkNodes(root, float('-inf'), float('inf'))

"""
99. Recover Binary Search Tree
"""
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        first = second = prev = None

        # Helper function
        def inorder(node):
            nonlocal first, second, prev
            if not node:
                return
            
            inorder(node.left)
            
            # Detect swapped nodes
            if prev and node.val < prev.val:
                if not first:  # First anomaly
                    first = prev
                second = node  # Second anomaly
            
            # Update node
            prev = node
            
            inorder(node.right)
        
        inorder(root)

        # Swap
        if first and second:
            first.val, second.val = second.val, first.val

"""
102. Binary Tree Level Order Traversal
"""
from collections import deque

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Base case: Empty list
        if not root:
            return []
        
        result = []
        queue = deque([root])

        # Iterate until all nodes are processed
        while queue:
            # Store all nodes at the same level
            length = len(queue)
            level = []

            for _ in range(length):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)

        return result

"""
103. Binary Tree Zigzag Level Order Traversal
"""
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Base case: Empty list
        if not root:
            return []
        
        result = []
        queue = deque([root])
        # Control the direction
        left_to_right = True

        while queue:
            length = len(queue)
            level = []

            for _ in range(length):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            if not left_to_right:
                level.reverse()
            
            result.append(level)
            left_to_right = not left_to_right
        
        return result

"""
105. Construct Binary Tree from Preorder and Inorder Traversal
"""
class Solution:
    def buildTree(self, preorder, inorder):
        # Create a hashmap for quick lookup of root index in inorder
        inorder_map = {val: idx for idx, val in enumerate(inorder)}
        preorder_index = 0

        def helper(left, right):
            nonlocal preorder_index
            # Base case: if there are no elements to construct the tree
            if left > right:
                return None
            
            # Pick the current root value from preorder traversal
            root_val = preorder[preorder_index]
            preorder_index += 1
            
            # Create the tree node
            root = TreeNode(root_val)
            
            # Recursively construct the left and right subtrees
            root.left = helper(left, inorder_map[root_val] - 1)
            root.right = helper(inorder_map[root_val] + 1, right)
            
            return root

        return helper(0, len(inorder) - 1)

"""

"""
