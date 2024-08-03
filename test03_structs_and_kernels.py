import pyopencl as pycl
import pyopencl as cl
import numpy as np
import unittest
import time
import pyopencl.cltypes
from pyopencl.array import Array as clArray
import pyopencl.clmath
import matplotlib.pyplot as plt

class TestOpenCLWithStructs(unittest.TestCase):
    """
    When writing computation kernels, it becomes useful to use structs to package
    the data properly.  The Python structs and c-structs are different
    and require some glue-code to makje it work properly.

    I am following the documentiaton here
    https://documen.tician.de/pyopencl/howto.html
    """

    def setUp(self):
        self.gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        self.context = pycl.Context(devices=self.gpuDevices)
        self.queue = pycl.CommandQueue(self.context)

    def test_01_accessing_clarrays(self):
        """
        We need to call .get() to get the numpy array. If not, we get an error.
        """
        ary_host = np.empty(20, float)
        ary_host.fill(123)

        ary = cl.array.to_device(self.queue, ary_host)

        for v in ary.get():
            self.assertTrue(v == 123.0)

    def test_02_declare_struct_register(self):
        """
        I am following the documentiaton here
        https://documen.tician.de/pyopencl/howto.html

        So we create a structure in numpy with its c-equivalent declaration that we will
        pass to the program.

        """

        my_struct = np.dtype([("field1", np.int32), ("field2", np.float32)])
        self.assertIsNotNone(my_struct)
        
        my_struct2, my_struct_c_decl = cl.tools.match_dtype_to_c_struct(self.gpuDevices[0], "my_struct", my_struct)
        self.assertEqual(my_struct, my_struct2)

        my_struct = cl.tools.get_or_register_dtype("my_struct", my_struct)

        ary_host = np.empty(20, my_struct)
        ary_host["field1"].fill(217)
        ary_host["field2"].fill(1000)
        ary_host[13]["field2"] = 12

        ary = cl.array.to_device(self.queue, ary_host)

        prg = cl.Program(self.context, my_struct_c_decl + """
                    __kernel void set_to_1(__global my_struct *a)
                    {
                        a[get_global_id(0)].field1 = 1;
                    }
                    """).build()

        evt = prg.set_to_1(self.queue, ary.shape, None, ary.data)

        for i, element in enumerate(ary.get()):
            self.assertEqual(element[0], 1)

            if i == 13:
                self.assertEqual(element[1], 12)
            else:
                self.assertEqual(element[1], 1000)

if __name__ == "__main__":
    unittest.main()
