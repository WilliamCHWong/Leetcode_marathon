from typing import List, Optional

"""
94. Binary Tree Inorder Traversal
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
100. Same Tree
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
    
"""
104. Maximum Depth of Binary Tree
"""
    # Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return left + 1 if left > right else right + 1
    
"""
101. Symmetric Tree
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def compare(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            
            return (left.val == right.val) and compare(left.left, right.right) and compare(left.right, right.left)
        return compare(root.left, root.right)
    
"""
108. Convert Sorted Array to Binary Search Tree
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
         def buildTree(left, right):
             if left > right:
                 return None
             mid = (left + right) // 2
             root = TreeNode(nums[mid])
             root.left = buildTree(left, mid - 1)
             root.right = buildTree(mid + 1, right)
             return root
         return buildTree(0, len(nums) - 1)
"""
110. Balanced Binary Tree
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def depth_and_balanced(root):
            if root is None:
                return 0, True
            left_depth, left_balanced = depth_and_balanced(root.left)
            if left_balanced is False:
                return 0, False
            right_depth, right_balanced = depth_and_balanced(root.right)

            current_depth = 1 + max(left_depth, right_depth)

            return current_depth, abs(left_depth - right_depth) <= 1 and left_balanced and right_balanced
        
        _, balanced = depth_and_balanced(root)
        return balanced
    
"""
111. Minimum Depth of Binary Tree
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        # Empty node
        if not root:
            return 0
        # One side child
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
    
"""
112. Path Sum
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
       
        # Check if it is a leaf 
        if not root.left and not root.right and targetSum == root.val:
            return True

        # Recursively check the left and right subtrees
        targetSum -= root.val
        return (self.hasPathSum(root.left, targetSum) or 
                self.hasPathSum(root.right, targetSum))
    
"""
144. Binary Tree Preorder Traversal
"""
    
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        result = [root.val]
        result.extend(self.preorderTraversal(root.left))
        result.extend(self.preorderTraversal(root.right))
        return result