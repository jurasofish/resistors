import mip
from typing import List


def equivalent_series(resistors: List[float], target: float) -> List[float]:
    """Return list of resistors which in series are closest to target resistance.

    Args:
        resistors: float values of the resistors to choose from. A resistor
            value can be used as many times as it occurs in this list.
        target: The target resistance.

    Returns:
        Optimal resistor values.
    """
    m = mip.Model()  # Create new mixed integer/linear model.

    # Will take value of 1 when corresponding resistor is in use, otherwise 0.
    r_in_use = [m.add_var(var_type=mip.BINARY) for _ in resistors]
    opt_r = sum([b * r for b, r in zip(r_in_use, resistors)])  # This will be the optimal resistance
    error = opt_r - target  # Want to minimise the absolute value of this error.

    # create a variable which is greater than than the absolute value of the error.
    # Because we will be minimizing, this will be forced down to equal the
    # absolute value. Common trick, google "linear programming absolute value".
    abs_error = m.add_var(lb=0)
    m += abs_error >= error
    m += abs_error >= -1 * error

    # Objective of the optimisation is to minimise the absolute error.
    m.objective = mip.minimize(abs_error)
    m.verbose = False  # Turn off verbose logging output.
    sol_status = m.optimize()
    assert sol_status == mip.OptimizationStatus.OPTIMAL

    # Get the solution values telling us which resistors are in use.
    r_in_use_sol = [float(v) for v in r_in_use]

    # Pick out the values of the resistors corresponding to the resistors
    # that the optimiser decided to use.
    r_to_use = [r for r, i in zip(resistors, r_in_use_sol) if i > 0]

    solved_resistance = sum(x for x in r_to_use)
    solved_error = 100 * (solved_resistance - target) / target
    print(f'Resistors {r_to_use} in series will produce '
          f'R={solved_resistance:.3f}. Aiming for R={target:.3f}, '
          f'error of {solved_error:.2f}%')
    return r_to_use


sol = equivalent_series([1, 2, 3, 4, 5, 6, 7], 11)
sol = equivalent_series([1, 2, 3, 4, 5, 6, 7], 15.6)
sol = equivalent_series(list(range(1, 100)), 1056)
