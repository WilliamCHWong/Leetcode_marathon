from typing import List

def search(nums: List[int], target: int) -> bool:
    n = len(nums) - 1

    i = 0
    while nums[i] <= nums[i + 1] and i  < n:
        i += 1

    def toIndex(k: int):
        return (k + i + 1) % n
    
    left, right = 0, n - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[toIndex(mid)] == target or nums[toIndex(left)] == target or nums[toIndex(right)] == target:
            return True
        
        elif nums[toIndex(mid)] < target:
            left = mid + 1
        
        else:
            right  = mid - 1
    
    return False

nums = [2,5,6,0,0,1,2]
target = 0

print(search(nums, target))