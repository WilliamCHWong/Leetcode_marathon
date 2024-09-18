# Two Sum

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        record = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in record:
                return [record[complement], i]
            record[num] = i
        return []
    