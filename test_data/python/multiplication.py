def multiplication(a float, b: float) -> float:
    return a * b

def division(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Can't divide by 0.")
    return a / b
