#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "legendre.h"

enum xnum_method {
    METHOD_PLM = 1,
    METHOD_XNUM = 2
};

static int parse_method(const char *value, enum xnum_method *method)
{
    if (strcmp(value, "plm") == 0) {
        *method = METHOD_PLM;
        return 1;
    }
    if (strcmp(value, "xnum") == 0) {
        *method = METHOD_XNUM;
        return 1;
    }

    PyErr_SetString(PyExc_ValueError, "method must be 'plm' or 'xnum'");
    return 0;
}

static int double_to_int(double value, const char *name, int *result)
{
    if (!isfinite(value) || value < (double)INT_MIN || value > (double)INT_MAX) {
        PyErr_Format(PyExc_ValueError, "%s must be a finite value in the C int range", name);
        return 0;
    }
    *result = (int)value;
    return 1;
}

static int py_number_to_int(PyObject *value, const char *name, int *result)
{
    double number = PyFloat_AsDouble(value);
    if (number == -1.0 && PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "%s must be a real scalar", name);
        return 0;
    }
    return double_to_int(number, name, result);
}

static int check_element_count(size_t first, size_t second)
{
    if (first != 0 && second > (size_t)INT_MAX / first) {
        PyErr_SetString(
            PyExc_OverflowError,
            "the requested output is too large for the underlying Legendre routines"
        );
        return 0;
    }
    return 1;
}

static void *allocate_doubles(size_t count)
{
    if (count == 0) {
        count = 1;
    }
    if (count > SIZE_MAX / sizeof(double)) {
        return NULL;
    }
    return malloc(count * sizeof(double));
}

static PyArrayObject *new_fortran_matrix(npy_intp rows, npy_intp columns)
{
    npy_intp dimensions[2] = {rows, columns};
    return (PyArrayObject *)PyArray_EMPTY(2, dimensions, NPY_DOUBLE, 1);
}

static PyObject *return_outputs(
    int derivatives,
    PyArrayObject *values,
    PyArrayObject *first,
    PyArrayObject *second
)
{
    PyObject *result;

    if (derivatives == 0) {
        return (PyObject *)values;
    }

    result = PyTuple_New(derivatives + 1);
    if (result == NULL) {
        Py_DECREF(values);
        Py_XDECREF(first);
        Py_XDECREF(second);
        return NULL;
    }

    PyTuple_SET_ITEM(result, 0, (PyObject *)values);
    PyTuple_SET_ITEM(result, 1, (PyObject *)first);
    if (derivatives == 2) {
        PyTuple_SET_ITEM(result, 2, (PyObject *)second);
    }
    return result;
}

static PyObject *py_assoc_legendre(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyObject *degree_object;
    PyObject *theta_object;
    PyObject *order_object = Py_None;
    const char *method_string = "plm";
    int speed_od = 0;
    int derivatives = 0;
    static char *keywords[] = {
        "degree", "theta", "order", "method", "speed_od", "derivatives", NULL
    };

    enum xnum_method method;
    PyArrayObject *degree_array = NULL;
    PyArrayObject *theta_array = NULL;
    PyArrayObject *values = NULL;
    PyArrayObject *first = NULL;
    PyArrayObject *second = NULL;
    double *degree;
    double *theta;
    double *values_data;
    double *first_data = NULL;
    double *second_data = NULL;
    double *Wmm = NULL;
    double *Wlm_1 = NULL;
    double *Wlm_2 = NULL;
    npy_intp degree_length_np;
    npy_intp theta_length_np;
    int degree_length;
    int theta_length;
    int lmax = 0;
    int order = 0;
    int numlp = 0;
    size_t coefficient_count = 0;
    size_t wmm_count = 0;
    size_t wlm1_count = 0;
    size_t wlm2_count = 0;
    int use_xnum;
    int computation_ok;
    int index;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OO|Ospi:assoc_legendre",
            keywords,
            &degree_object,
            &theta_object,
            &order_object,
            &method_string,
            &speed_od,
            &derivatives)) {
        return NULL;
    }

    if (!parse_method(method_string, &method)) {
        return NULL;
    }
    if (derivatives < 0 || derivatives > 2) {
        PyErr_SetString(PyExc_ValueError, "derivatives must be 0, 1, or 2");
        return NULL;
    }

    degree_array = (PyArrayObject *)PyArray_FROM_OTF(
        degree_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    theta_array = (PyArrayObject *)PyArray_FROM_OTF(
        theta_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (degree_array == NULL || theta_array == NULL) {
        goto fail;
    }

    degree_length_np = PyArray_SIZE(degree_array);
    theta_length_np = PyArray_SIZE(theta_array);
    if (degree_length_np > INT_MAX || theta_length_np > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "input vectors are too large");
        goto fail;
    }

    degree_length = (int)degree_length_np;
    theta_length = (int)theta_length_np;
    degree = (double *)PyArray_DATA(degree_array);
    theta = (double *)PyArray_DATA(theta_array);

    if (speed_od) {
        if (degree_length != 1) {
            PyErr_SetString(
                PyExc_ValueError,
                "degree must be a scalar Lmax when speed_od=True"
            );
            goto fail;
        }
        if (order_object != Py_None) {
            PyErr_SetString(
                PyExc_ValueError,
                "order is not used when speed_od=True"
            );
            goto fail;
        }
        if (!double_to_int(degree[0], "Lmax", &lmax)) {
            goto fail;
        }
        if (lmax < 0) {
            PyErr_SetString(PyExc_ValueError, "Lmax must be non-negative");
            goto fail;
        }

        coefficient_count = ((size_t)lmax + 1) * ((size_t)lmax + 2) / 2;
        if (coefficient_count > INT_MAX ||
            !check_element_count(coefficient_count, (size_t)theta_length)) {
            goto fail;
        }

        values = new_fortran_matrix((npy_intp)coefficient_count, theta_length_np);
        if (derivatives >= 1) {
            first = new_fortran_matrix((npy_intp)coefficient_count, theta_length_np);
        }
        if (derivatives == 2) {
            second = new_fortran_matrix((npy_intp)coefficient_count, theta_length_np);
        }

        wmm_count = (size_t)lmax;
        wlm1_count = (size_t)lmax * ((size_t)lmax + 1) / 2;
        wlm2_count = lmax > 1 ? (size_t)lmax * ((size_t)lmax - 1) / 2 : 0;
    } else {
        for (index = 0; index < degree_length; ++index) {
            int current_degree;
            if (!double_to_int(degree[index], "degree", &current_degree)) {
                goto fail;
            }
            if (current_degree > lmax) {
                lmax = current_degree;
            }
        }

        if (order_object != Py_None && !py_number_to_int(order_object, "order", &order)) {
            goto fail;
        }
        if (order > lmax) {
            order = lmax;
        }
        if (order < 0) {
            order = 0;
        }

        numlp = lmax - order + 1;
        if (!check_element_count((size_t)degree_length, (size_t)theta_length)) {
            goto fail;
        }

        values = new_fortran_matrix(theta_length_np, degree_length_np);
        if (derivatives >= 1) {
            first = new_fortran_matrix(theta_length_np, degree_length_np);
        }
        if (derivatives == 2) {
            second = new_fortran_matrix(theta_length_np, degree_length_np);
        }

        wmm_count = (size_t)order + 1;
        wlm1_count = (size_t)numlp;
        wlm2_count = numlp > 1 ? (size_t)(numlp - 1) : 0;
    }

    if (values == NULL || (derivatives >= 1 && first == NULL) ||
        (derivatives == 2 && second == NULL)) {
        goto fail;
    }

    Wmm = (double *)allocate_doubles(wmm_count);
    Wlm_1 = (double *)allocate_doubles(wlm1_count);
    Wlm_2 = (double *)allocate_doubles(wlm2_count);
    if (Wmm == NULL || Wlm_1 == NULL || Wlm_2 == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    values_data = (double *)PyArray_DATA(values);
    if (first != NULL) {
        first_data = (double *)PyArray_DATA(first);
    }
    if (second != NULL) {
        second_data = (double *)PyArray_DATA(second);
    }

    Py_BEGIN_ALLOW_THREADS
    use_xnum = method == METHOD_XNUM;

    if (speed_od) {
        init_legendre_1st_kind_od_speed(lmax, Wmm, Wlm_1, Wlm_2);
        if (use_xnum) {
            computation_ok = legendre_1st_kind_od_xnum_speed(
                lmax, theta_length, theta, values_data, first_data, second_data,
                Wmm, Wlm_1, Wlm_2
            );
        } else {
            computation_ok = legendre_1st_kind_od_speed(
                lmax, theta_length, theta, values_data, first_data, second_data,
                Wmm, Wlm_1, Wlm_2
            );
        }
    } else if (use_xnum) {
        init_legendre_1st_kind_plm(lmax, order, Wmm, Wlm_1, Wlm_2);
        computation_ok = legendre_1st_kind_xnum_plm(
            lmax, order, numlp, degree_length, degree,
            theta_length, theta, values_data, first_data, second_data,
            Wmm, Wlm_1, Wlm_2
        );
    } else {
        init_legendre_1st_kind_plm(lmax, order, Wmm, Wlm_1, Wlm_2);
        computation_ok = legendre_1st_kind_plm(
            lmax, order, numlp, degree_length, degree,
            theta_length, theta, values_data, first_data, second_data,
            Wmm, Wlm_1, Wlm_2
        );
    }
    Py_END_ALLOW_THREADS

    free(Wmm);
    free(Wlm_1);
    free(Wlm_2);
    Py_DECREF(degree_array);
    Py_DECREF(theta_array);

    if (!computation_ok) {
        Py_DECREF(values);
        Py_XDECREF(first);
        Py_XDECREF(second);
        PyErr_SetString(PyExc_RuntimeError, "the Legendre calculation failed");
        return NULL;
    }

    return return_outputs(derivatives, values, first, second);

fail:
    free(Wmm);
    free(Wlm_1);
    free(Wlm_2);
    Py_XDECREF(degree_array);
    Py_XDECREF(theta_array);
    Py_XDECREF(values);
    Py_XDECREF(first);
    Py_XDECREF(second);
    return NULL;
}

PyDoc_STRVAR(
    assoc_legendre_doc,
    "assoc_legendre(degree, theta, order=None, method='plm', speed_od=False, derivatives=0)\n"
    "--\n\n"
    "Compute fully normalized associated Legendre functions.\n\n"
    "Parameters\n"
    "----------\n"
    "degree : array_like or scalar\n"
    "    Degrees in PLM mode, or scalar Lmax when speed_od=True. Values are\n"
    "    converted to C integers exactly as in the MEX implementation.\n"
    "theta : array_like\n"
    "    Co-latitudes in radians.\n"
    "order : scalar, optional\n"
    "    Associated order for PLM mode. Defaults to zero.\n"
    "method : {'plm', 'xnum'}, optional\n"
    "    Use ordinary or X-number-stabilized computation independently of\n"
    "    the output mode.\n"
    "speed_od : bool, optional\n"
    "    If true, calculate every degree/order pair through scalar Lmax.\n"
    "derivatives : {0, 1, 2}, optional\n"
    "    Highest derivative to compute. Zero returns P; one returns (P, dP);\n"
    "    two returns (P, dP, ddP).\n\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray or tuple of numpy.ndarray\n"
    "    Fortran-contiguous float64 arrays with the same shapes and ordering\n"
    "    as the MATLAB MEX outputs. When speed_od=True, rows are ordered as\n"
    "    00, 10, 20, ..., 11, 21, ..., 22, ...."
);

static PyMethodDef module_methods[] = {
    {
        "assoc_legendre",
        (PyCFunction)(void (*)(void))py_assoc_legendre,
        METH_VARARGS | METH_KEYWORDS,
        assoc_legendre_doc
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "legendre_shbundle",
    "NumPy interface to SHBundle's native Legendre routines.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_legendre_shbundle(void)
{
    PyObject *module;

    import_array();
    module = PyModule_Create(&module_definition);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
