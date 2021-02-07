import mip


def equivalent_resistors(resistors, target):

    resistors_recip = [1/x for x in resistors]
    target_recip = 1/target

    m = mip.Model()  # Create new mixed integer/linear model.

    # Will take value of 1 when corresponding resistor is in use, otherwise 0.
    r_in_use = [m.add_var(var_type=mip.BINARY) for _ in resistors_recip]
    opt_r = sum([b * r for b, r in zip(r_in_use, resistors_recip)])  # This will be the optimal resistance
    error = opt_r - target_recip  # Want to minimise the absolute value of this error.

    # create a variable which is greater than than the absolute value of the error.
    # Because we will be minimizing, this will be forced down to equal the
    # absolute value. Common trick, google "linear programming absolute value".
    abs_error = m.add_var(lb=0)
    m += abs_error >= error
    m += abs_error >= -1 * error

    # Objective of the optimisation is to minimise the absolute error.
    m.objective = mip.minimize(abs_eror)
    m.verbose = False  # Turn off verbose logging output.
    sol_status = m.optimize()
    assert sol_status == mip.OptimizationStatus.OPTIMAL

    # Get the solution values telling us which resistors are in use.
    r_in_use_sol = [float(v) for v in r_in_use]

    # Pick out the values of the resistors corresponding to the resistors
    # that the optimiser decided to use.
    r_to_use = [r for r, i in zip(resistors, r_in_use_sol) if i > 0]

    solved_resistance = 1/sum(1/x for x in r_to_use)
    solved_error = 100 * (solved_resistance - target) / target
    print(f'Resistors {r_to_use} in parallel will produce '
          f'R={solved_resistance:.3f}. '
          f'Aiming for R={target:.3f}, '
          f'error of {solved_error:.2f}%')
    return r_to_use


def main():
    print(f'mip version {mip.version}')
    sol = equivalent_resistors([1, 2, 3, 4, 5, 6, 7], 1.5555)
    sol = equivalent_resistors([1, 2, 3, 4, 5, 6, 7], 1.9)
    sol = equivalent_resistors(list(range(1, 100)), 123)
    sol = equivalent_resistors(list(range(1, 1000)), 5.954520294)
    sol = equivalent_resistors(list(range(1, 10_000)), 5.954520294)


if __name__ == '__main__':
    main()
