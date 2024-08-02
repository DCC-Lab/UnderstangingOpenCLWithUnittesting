import pyopencl as pycl
import pyopencl as cl
import numpy as np
import unittest
import time
import pyopencl.cltypes
from pyopencl.array import Array as clArray
import pyopencl.clmath
import matplotlib.pyplot as plt

class TestOpenCLIds(unittest.TestCase):
    """
    I am now starting to write my own kernels.  I found it complicated to understand
    global_id, local_id, etc.... So this series of tests is to understand these workunit functions
    (that's what they are called). It is necessary to access the proper data in the computation.
    """

    kernel_source_file = """
    // ----------------- TEST KERNELS -----------------

     __kernel void test_global_id(__global int *buffer){
        int id = get_global_id(0);
        buffer[id] = id;
    }

     __kernel void test_local_id(__global int *buffer){
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        buffer[gid] = lid;
    }


     __kernel void test_extract_many_ids(__global int *buffer){
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        int lgid = get_group_id(0);
        int ls = get_local_size(0);
        int offset = get_global_offset(0);
        buffer[gid] = gid+lid*100+ls*10000+lgid*1000000+offset*100000000;
    }

     __kernel void test_compute_global_id_from_local_id(__global int *buffer){
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        int lgid = get_group_id(0);
        int ls = get_local_size(0);
        int offset = get_global_offset(0);
        buffer[gid] = ls * lgid + lid ;
    }

     __kernel void test_compute_global_id_from_local_id_nonuniform(__global int *buffer, __global int *local_sizes){
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        int lgid = get_group_id(0);
        int ls = get_local_size(0);
        local_sizes[lgid] = ls;
        int i = 0;
        int total_previous = 0;
        for (i = 0; i < lgid; i++) {
            total_previous += local_sizes[i];
        }
        int offset = get_global_offset(0);
        buffer[gid] = total_previous + lid + offset;
    }
    """

    def setUp(self):
        self.gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        self.context = pycl.Context(devices=self.gpuDevices)
        self.queue = pycl.CommandQueue(self.context)
        self.program_source = self.kernel_source_file
        self.program = pycl.Program(self.context, self.program_source).build()

    def test_01_get_kernel_info(self):
        """
        Can I get the kernel from my source file above? See setUp()
        """
        kernel = self.program.test_global_id
        self.assertIsNotNone(kernel)

    def test_02_trivial_kernel_global_id(self):
        """
        global_id is a number from 0 to nWorkUnits-1 that is associated with each "kernel run"
        A kernel run is applied on a "work unit".

        My first kernel called 'test_global_id' is the variable kernel_source_file above
        will simply copy the global_id into the array that I am passing to it.

        I will overwrite the default kernel file by overwriting it.
        """
        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 128
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        self.program_source = """
             __kernel void test_global_id(__global uint *buffer){
                int id = get_global_id(0);
                buffer[id] = id;
            }
        """
        self.program = pycl.Program(self.context, self.program_source).build()

        knl = self.program.test_global_id # copy global_id into array

        """
        Looking at knl below and the kernel source file first line:
                __kernel void test_global_id(__global uint *buffer)

        queue: is the opencl queue
        nWorkUnits:  is the total number of computations. We have kernel_source_file elements.
        The second parameter (None) is the group size because the whole computation is
        split into group_size "local" computations. We let OpenCL (i.e. it should be 32)
        valueBuffer.data : is the array passed to the kernel.
        """

        knl(queue, (nWorkUnits,), None, valueBuffer.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(value, i)

    def test_03_trivial_kernel_local_id_default_workgroup_size(self):
        """
        local_id is a number that I don't really know what it does.
        Not sure when to use it, but all workunits are split into groups, of get_group_size()

        I will force the group_size to see if this works (knl third parameter)

        """
        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 128
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        self.program_source = """
             __kernel void test_local_id(__global int *buffer){
                int gid = get_global_id(0);
                int lid = get_local_id(0);
                buffer[gid] = lid;
            }
        """
        self.program = pycl.Program(self.context, self.program_source).build()

        knl = self.program.test_local_id # copy local_id into array

        local_group_size = 32 
        knl(queue, (nWorkUnits,), (local_group_size,), valueBuffer.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(i%local_group_size, value)

    def test_04_trivial_kernel_local_id_set_workgroup_size(self):
        """
        Having attempted to set the workgroup size, I noticed not all values are possible.

        The number of groups must divide nWorkUnits evenly
        """

        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 100
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_local_id # copy local_id into array

        accepted_values = set()
        for i in range(100):
            try:
                knl(queue, (nWorkUnits,), (i,), valueBuffer.data)
                self.assertEqual(nWorkUnits%i, 0)
                accepted_values.add(i)
                for j, value in enumerate(valueBuffer):
                    self.assertEqual(j%nWorkUnits, value)
            except:
                pass

        self.assertTrue(accepted_values == set([1,2,4,5,10,20,25,50]))

    # @unittest.skip('For information only')
    # def test_05_extract_many_ids(self):
    #     nWorkUnits = 100
    #     valueBuffer = Buffer(nWorkUnits, value=0, dtype=cl.cltypes.uint)

    #     source_path = os.path.join(OPENCL_SOURCE_DIR, "test_opencl.c")
    #     program = CLProgram(source_path)
    #     program.launchKernel(kernelName='test_extract_many_ids', N=nWorkUnits, arguments = [valueBuffer])
    #     for i, value in enumerate(valueBuffer.hostBuffer):
    #         print(i,value)

    def test_06_compute_global_id_from_local_id_uniform_sizes(self):
        """
        When we force a uniform group size, it is possible to compute the global_id from the local_id.
        """
        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 128 # Multiplie of 32
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_compute_global_id_from_local_id # copy local_id into array
        knl(queue, (nWorkUnits,), None, valueBuffer.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(value, i)

    def test_07_compute_global_id_from_local_id_NON_uniform_sizes(self):
        """
        Computing the global_id from the local_id in a non-uniform computation
        is problematic, because we need to know the size of all groups before the present
        group.  This is not guaranteed (as this tests demonstrate).

        My assumption was that workunits would be fed in order but that is not true.
        I don't know how to compute the global_id from the local parameters then
        It does not matter: use get_global_id() but this is a basic issue in 
        understanding OpenCL for me.
        """

        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 100 # NOT A MULTIPLE of 32
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)
        local_sizes = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_compute_global_id_from_local_id_nonuniform # copy local_id into array
        knl(queue, (nWorkUnits,), None, valueBuffer.data, local_sizes.data)

        # This is fine: IN THE END, local_sizes is correct
        self.assertEqual(np.sum(local_sizes), nWorkUnits)

        # But this is not: the calculated global_id is incorrect
        # because during the computation, local_sizes is not completed in order.
        with self.assertRaises(Exception):
            for i, value in enumerate(valueBuffer):
                self.assertEqual(value, i)

if __name__ == "__main__":
    unittest.main()
