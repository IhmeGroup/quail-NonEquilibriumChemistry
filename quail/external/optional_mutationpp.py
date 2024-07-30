# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2024
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/external/optional_mutationpp.py
#
#       Contains mock class definition for Mutation++ module
#
# ------------------------------------------------------------------------ #
try:
	import mutationpp as mpp

except ImportError:
	class MutationppMock():
		'''
		Defines a mock class for cantera. This ensures that users
		do not need to have cantera for quail to run successfully
		'''
		def __init__(self):
			self.one_atm = 1.
			return
		def __repr__(self):
			return 'Warning: {self.__class__.__name__} is a mock class' \
				.format(self=self)
	mpp = MutationppMock()