import numpy as np
import pytest
import os
import pickle
import subprocess

import list_of_cases
import quail.physics.euler.euler as euler
import quail.physics.scalar.scalar as scalar
import quail.physics.chemistry.chemistry as chemistry
import quail.solver.DG as DG
import quail.solver.ADERDG as ADERDG

# Tolerances

# Markers distinguish tests into different categories
@pytest.mark.e2e
def test_case(test_data):
	'''
	This function runs Quail by calling the executable with an input file and
	asserts that the result matches the output from previous versions of the
	code by comparing with a previously generated regression data file. The
	full vector of solution coefficients is compared.

	Inputs:
	-------
		test_data: Pytest fixture containing the expected solution as well
			as the root Quail directory
	'''

	# Unpack test data
	Uc_expected_list, quail_dir = test_data

	# Get case directory
	case_dir = os.path.dirname(os.path.abspath(__file__))
	# Get test case name
	test_name = case_dir.split('cases/')[-1]

	# Get expected solution for this test case
	Uc_expected = Uc_expected_list[test_name]

	# Enter case directory
	os.chdir(case_dir)

	# Call the Quail executable
	result = subprocess.check_output([
			f'{quail_dir}/quail/main.py', 'input_file.py',
			], stderr=subprocess.STDOUT)
	# Print results of run
	print(result.decode('utf-8'))

	# Open resulting data file
	with open('Data_final.pkl', 'rb') as f:
		# Read final solution from file
		solver = pickle.load(f)
		Uc = solver.state_coeffs
		# Assert
		np.testing.assert_allclose(Uc, Uc_expected, list_of_cases.rtol, list_of_cases.atol)
