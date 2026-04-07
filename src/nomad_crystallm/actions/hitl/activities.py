from temporalio import activity


@activity.defn
def generate_random_number_activity(lower_bound: int, upper_bound: int) -> int:
    """
    Generate a random integer between lower_bound and upper_bound (inclusive).
    """
    import random

    if lower_bound > upper_bound:
        raise ValueError('lower_bound must be less than or equal to upper_bound')

    return random.randint(lower_bound, upper_bound)
