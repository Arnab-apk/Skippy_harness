"""A simple Python test module."""


def greet(name: str) -> str:
    """Return a greeting message for the given name.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting string
    """
    return f"Hello, {name}!"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b


def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers together.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Result of division
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def get_max(nums):
    """Find the maximum value in a list of numbers.
    
    Args:
        nums: A list of numbers
        
    Returns:
        The maximum number in the list
    """
    if not nums:
        return None
    
    max_num = nums[0]
    for num in nums:
        if num > max_num:
            max_num = num
    
    return max_num


if __name__ == "__main__":
    # Run tests
    print("Testing greet function:")
    print(f"  greet('World') = {greet('World')}")
    
    print("\nTesting add_numbers function:")
    print(f"  add_numbers(5, 3) = {add_numbers(5, 3)}")
    
    print("\nTesting divide_numbers function:")
    result = divide_numbers(10, 2)
    print(f"  divide_numbers(10, 2) = {result}")
    
    print("\nTesting get_max function:")
    nums = [5, 3, 8, 1, 9]
    max_num = get_max(nums)
    print(f"  max([5, 3, 8, 1, 9]) = {max_num}")
