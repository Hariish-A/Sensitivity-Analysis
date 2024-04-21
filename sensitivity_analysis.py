import numpy as np

def simplex(c, A, b):
    m, n = A.shape

    # Add slack variables to convert inequalities to equalities
    A_slack = np.hstack([A, np.eye(m)])
    c_slack = np.concatenate([c, np.zeros(m)])
    basic_indices = np.arange(n, n + m)

    # Phase 1: Find a basic feasible solution
    c_phase1 = np.concatenate([np.zeros(n), np.ones(m)])
    res = simplex_solve(c_phase1, A_slack, b, basic_indices)

    if res['success']:
        basic_indices = res['basic_indices']
        x = res['x']
        B_inv = res['B_inv']
        if np.any(x[n:]) > 0:
            print("Phase 1 found a basic feasible solution")
        else:
            print("No feasible solution exists")
            return None
    else:
        print("Phase 1 failed")
        return None

    # Phase 2: Solve the original problem
    res = simplex_solve(c_slack, A_slack, b, basic_indices, B_inv)

    if res['success']:
        print("Optimal solution found:", res['x'][:n])
        print("Optimal objective value:", res['objective'])
        print("Shadow prices (dual variables):", res['dual_vars'])
        print("Reduced costs:", res['reduced_costs'])
        print("Slack/Surplus values:", res['slack'])
    else:
        print("Phase 2 failed")

def simplex_solve(c, A, b, basic_indices, B_inv=None):
    m, n = A.shape

    if B_inv is None:
        B_inv = np.linalg.inv(A[:, basic_indices])

    while True:
        c_b = c[basic_indices]
        B_inv = np.linalg.inv(A[:, basic_indices])

        dual_vars = c_b.dot(B_inv)
        reduced_costs = c - A.T.dot(dual_vars)

        if np.all(reduced_costs <= 0):
            x = np.zeros(n)
            x[basic_indices] = B_inv.dot(b)
            objective = c.dot(x)
            slack = b - A.dot(x)
            return {'success': True, 'x': x, 'basic_indices': basic_indices, 'objective': objective,
                    'dual_vars': dual_vars, 'reduced_costs': reduced_costs, 'slack': slack, 'B_inv': B_inv}

        entering_var = np.argmax(reduced_costs)
        d = np.linalg.solve(B_inv, A[:, entering_var])

        if np.all(d <= 0):
            return {'success': False}

        ratios = np.divide(b, d, out=np.full_like(b, np.inf), where=d > 0)
        leaving_var = np.argmin(ratios)

        basic_indices[leaving_var] = entering_var

if __name__ == "__main__":
    # Get input from the user for the objective function coefficients
    c = [float(x) for x in input("Enter the coefficients of the objective function separated by spaces: ").split()]

    # Get input from the user for the coefficients of the constraints
    num_constraints = int(input("Enter the number of constraints: "))
    A = []
    b = []
    for i in range(num_constraints):
        constraint_coeffs = [float(x) for x in input(f"Enter the coefficients of constraint {i+1} separated by spaces: ").split()]
        A.append(constraint_coeffs[:-1])
        b.append(constraint_coeffs[-1])

    A = np.array(A)
    b = np.array(b)

    simplex(np.array(c), A, b)