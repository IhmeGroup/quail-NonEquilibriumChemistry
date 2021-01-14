import os
import subprocess
import sys
sys.path.append('../../src')

import numpy as np

import list_of_cases


def generate_tests():

    # Get script directory
    script_dir = sys.path[0]

    # Add full path to case directories
    case_dirs = [f'{script_dir}/{case_dir}' for case_dir in
            list_of_cases.case_dirs.keys()]
    markers = list(list_of_cases.case_dirs.values())
    n_cases = len(case_dirs)

    # Read in base test script
    with open ("base_test_script.py", "r") as base_test_file:
        base_test = base_test_file.readlines()

    # Find which line to add markers to
    line_of_markers = 0
    for i, line in enumerate(base_test):
        if line.startswith('# Markers'):
            line_of_markers = i + 1
            break

    # Loop over all case directories
    for i, (case_dir, marker_list) in enumerate(zip(case_dirs, markers)):
        # Move to the test case directory
        os.chdir(case_dir)

        # Add markers to test script
        test_script = base_test.copy()
        for marker in marker_list:
            test_script.insert(line_of_markers, f'@pytest.mark.{marker}\n')

        # Create test script
        with open(f'test_case_{i}.py', 'w') as test_case_file:
            for line in test_script:
                test_case_file.write(line)


if __name__ == "__main__":
    generate_tests()
