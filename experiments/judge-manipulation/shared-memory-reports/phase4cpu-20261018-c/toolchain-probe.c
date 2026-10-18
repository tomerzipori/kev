#include <Python.h>
static PyObject *value(PyObject *self, PyObject *args) {
    return PyLong_FromLong(42);
}
static PyMethodDef methods[] = {
    {"value", value, METH_NOARGS, "return a compile/link probe constant"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "kev_toolchain_probe", NULL, -1, methods
};
PyMODINIT_FUNC PyInit_kev_toolchain_probe(void) {
    return PyModule_Create(&module);
}
