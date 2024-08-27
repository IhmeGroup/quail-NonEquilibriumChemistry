# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/backend.py
#
#       Utilizes autoray (if available) to handle swapping different
#       computational backends for array computations.
#
# ------------------------------------------------------------------------ #
import numpy

try:
    import autoray

    set_backend = autoray.set_backend
    get_backend = autoray.get_backend
    set_backend('numpy')
    np = autoray.numpy
    autojit = autoray.autojit

    np.__dict__['add'] = autoray.autoray.NumpyMimic("add")

    def numpy_add_at(a, indices, b):
        """Generalized in-place addition operation.
        """
        numpy.add.at(a, indices, b)
        return a

    autoray.register_function('numpy', 'add.at', numpy_add_at)

    def jax_add_at(a, indices, b):
        a = a.at[indices].add(b.at[indices])
        return a

    autoray.register_function('jax', 'add.at', jax_add_at)

    try:
        # Add cunumeric backend
        import cunumeric

        autoray.register_backend(cls=cunumeric.ndarray, name="cunumeric")

        def cunumeric_add_at(a, indices, b):
            # Doesn't work when b.ndim > 1
            # a[:] += cunumeric.bincount(indices, weights=b,
            #                            minlength=a.shape[0])

            # Fall back to numpy
            numpy.add.at(a, indices, b)
            return a

        autoray.register_function('cunumeric', 'add.at', cunumeric_add_at)
    except:
        pass


except ImportError:
    np = numpy

    # Create mock decorator for just-in-time compilation which does nothing
    def autojit(func, *args, **kwargs):

        def do_nothing(*args, **kwargs):
            return func(*args, **kwargs)

        return do_nothing

    # Create mock functions for changing the backend
    def set_backend(like: str, get_globally: str | bool = 'auto') -> None:
        return

    def get_backend(get_globally: str | bool = 'auto') -> str:
        return 'numpy'
