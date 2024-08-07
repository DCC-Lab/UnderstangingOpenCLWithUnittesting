import pyopencl as pycl
import pyopencl as cl
import numpy as np
import unittest
import time
import pyopencl.cltypes
from pyopencl.array import Array as clArray
import pyopencl.clmath
import matplotlib.pyplot as plt

class TestOpenCL(unittest.TestCase):

    def test01Import(self):
        """
        Is it installed?
        """
        self.assertIsNotNone(pycl)

    def test02AtLeast1(self):
        """
        Let's get the platform so we can get the devices.
        There should be at least one.
        """
        self.assertTrue(len(pycl.get_platforms()) > 0)

    def test03AtLeast1Device(self):
        """
        Let's get the devices (graphics card?).  There could be a few.
        """
        platform = pycl.get_platforms()[0]
        devices = platform.get_devices()
        self.assertTrue(len(devices) > 0)

    def test031AtLeastGPUorCPU(self):
        devices = pycl.get_platforms()[0].get_devices()
        for device in devices:
            self.assertTrue(device.type == pycl.device_type.GPU or device.type == pycl.device_type.CPU)

    def test04Context(self):
        """ Finally, we need the context for computation context, requesting the GPU device if possible.
        """
        devices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        context = pycl.Context(devices=devices)
        self.assertIsNotNone(context)

    def test05GPUDevice(self):
        """
        There may be more than one GPU on the computer.  Originally, my computer had two.
        The internal Intel GPU card and the AMD GPU. The intel was low performance.  
        """
        gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        self.assertTrue(len(gpuDevices) >= 1)

        gpuDevice = [ device for device in gpuDevices if device.vendor == 'AMD']
        if gpuDevice is None:
            gpuDevice = [ device for device in gpuDevices if device.vendor == 'Intel']
        self.assertIsNotNone(gpuDevice)

    def test06ProgramSource(self):
        """
        Here is my first kernel.  I am following instructions directly from PyOpenCL
        """
        gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        context = pycl.Context(devices=gpuDevices)
        queue = pycl.CommandQueue(context)

        program_source = """
        kernel void sum(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] + b[gid];
                      }

        kernel void multiply(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] * b[gid];
                      }
        """
        program = pycl.Program(context, program_source).build()
        self.assertIsNotNone(program)

    def setUp(self):
        """
        Now that I know more, this will become handy to avoid repeating the
        code (it was already running all the time, but I will start using it
        from test07)
        """
        self.gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        self.context = pycl.Context(devices=self.gpuDevices)

    def test07CopyingBuffersFromHostToGPU(self):
        """
        I have a Kernel written, but I don't know how to pass an array to my kernel.
        I will start using the pycl.Buffer from PyOpenCL. This is not ioptimal (as I will
        see later, when I tsrta using pcl.Array), but this is the most general.

        My goal is to run my very simple kernel to see if this works.
        I will run the sum kernel code and multiply kernel code to sum and multiply 
        two arrays.

        Note:
        If I run this several times, I sometimes get 1 ms or 1000 ms.
        I suspect there is some stsartup time for PyOpenCL.
        I will write a setup function for the test

        I did, it is now much more stable at 1-2 ms.
        """

        # _np for numpy and _g for gpu
        N = 10000000
        a_np = np.random.rand(N).astype(np.float32)
        b_np = np.random.rand(N).astype(np.float32)
        self.assertIsNotNone(a_np)
        self.assertIsNotNone(b_np)

        mf = pycl.mem_flags
        a_g = pycl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = pycl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        self.assertIsNotNone(a_g)
        self.assertIsNotNone(b_g)
        res_g = pycl.Buffer(self.context, mf.WRITE_ONLY, a_np.nbytes)
        self.assertIsNotNone(res_g)


        queue = pycl.CommandQueue(self.context)

        program_source = """
        kernel void sum(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] + b[gid];
                      }

        kernel void multiply(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] * b[gid];
                      }
        """
        
        program = pycl.Program(self.context, program_source).build()

        # The build command makes the kernel function accessible as properties
        knlSum = program.sum        # Use this Kernel object for repeated calls
        knlProd = program.multiply  # Use this Kernel object for repeated calls

        for i in range(10):
            startTime = time.time()
            knlSum(queue, a_np.shape, None, a_g, b_g, res_g)
            knlProd(queue, a_np.shape, None, res_g, b_g, res_g)
            calcTime = time.time()-startTime
            res_np = np.empty_like(a_np)
            pycl.enqueue_copy(queue, res_np, res_g)
            copyTime = time.time()-startTime
            #print("\nCalculation time {0:.1f} ms, with copy {1:.1f} ms".format( 1000*calcTime, 1000*copyTime))

        # Check on CPU with Numpy:
        startTime = time.time()
        answer = (a_np + b_np)*b_np
        npTime = time.time() - startTime
        # print("Numpy {0:0.1f} ms".format(1000*npTime))
        assert np.allclose(res_np, answer)

    def test08NumpyVectorsWithOpenCLLayout(self):
        """
        In order to use pycl.Array, we need to use the cltype in numpy.
        """
        vectors = np.empty((128,), dtype=pycl.cltypes.float) # array of float
        self.assertIsNotNone(vectors)

    def test09CreateOpenCLArray(self):
        """
        clArray is optimized by the creators of PyOpenCL and should be sufficiently powerful
        for simple things like what I want (or at least to get started)
        """
        queue = pycl.CommandQueue(self.context)

        a = clArray(cq=queue, shape=(1024,), dtype=pycl.cltypes.float)
        self.assertIsNotNone(a)
        self.assertIsNotNone(a.data)

    def test10CreateOpenCLArray_and_initialize(self):
        """
        We can use the clArray like an array and assign values.
        """

        queue = pycl.CommandQueue(self.context)

        a = clArray(cq=queue, shape=(1024,), dtype=pycl.cltypes.float)
        self.assertIsNotNone(a)
        self.assertIsNotNone(a.data)
        for i in range(a.size):
            a[i] = i

        for i, e in enumerate(a):
            self.assertEqual(e, i)

    def test11_ManualScalarMultiplicationOfOpenCLArrays(self):
        """
        clArray provides simple operaators to work with arrays easily.
        """
        queue = pycl.CommandQueue(self.context)

        a = clArray(cq=queue, shape=(1024,), dtype=pycl.cltypes.float)
        self.assertIsNotNone(a)
        self.assertIsNotNone(a.data)

        for i in range(a.size):
            a[i] = i

        b = 2*a # This is done on the GPU thanks to the __mul__ operator of clArray

        for i, e in enumerate(b):
            self.assertEqual(e, 2*i)

    @unittest.skip('This is not testing much, it is mostly for informaiton purposes')
    def test12_ScalarMultiplicationOfOpenCLArrays(self):
        """
        As a curiosity, I test to see which one is faster: numpy or openCL.
        I suspect that OpenCL will be better but only with very large arrays.
        Also, the second time is faster (probably some init stuff already done)
        """
        queue = pycl.CommandQueue(self.context)
        a = clArray(cq=queue, shape=(2<<14,), dtype=pycl.cltypes.float)
        for i in range(a.size):
            a[i] = i

        startTime = time.time()        
        b = a+a
        calcTime = (time.time()-startTime)*1000
        print("\nOpenCL 1 scalar: {0:.1f} ms ".format(calcTime))

        a = np.array(object=[0]*(2<<14), dtype=pycl.cltypes.float)
        for i in range(a.size):
            a[i] = i

        startTime = time.time()        
        b = a+a
        calcTime = (time.time()-startTime)*1000
        print("\nnumpy: {0:.1f} ms ".format(calcTime))
        # for i, v in enumerate(b):
        #     self.assertEqual(v, 2*i)

        a = clArray(cq=queue, shape=(2<<14,), dtype=pycl.cltypes.float)
        for i in range(a.size):
            a[i] = i

        startTime = time.time()        
        b = a+a
        calcTime = (time.time()-startTime)*1000
        print("\nOpenCL 2 scalar: {0:.1f} ms ".format(calcTime))

    def test13ArraysWithAllocator(self):
        """
        I really expected this to work.  Performance is more complicated than I expected.
        The OpenCL calculation is much slower than the numpy version regardless of parameters 
        I used.

        The plan was simple: manipulate arrays in numpy and opencl, show it is much faster in Opencl.
        Well, it is not.

        UPDATE: yes it is, with VERY large arrays (2^18 or more). See next test.

        """

        # Set up basic OpenCl things
        queue = pycl.CommandQueue(self.context)
        allocator = pycl.tools.ImmediateAllocator(queue)
        mempool = pycl.tools.MemoryPool(allocator)

        N = 1<<18
        M = 1000

        # Pre-allocate all arrays
        a_n = np.random.rand(N).astype(np.float32)
        b_n = np.random.rand(N).astype(np.float32)

        # Pre-allocate opencl arrays with MemoryPool to reuse memory
        a = pycl.array.to_device(queue=queue, ary=a_n, allocator=mempool)
        b = pycl.array.to_device(queue=queue, ary=b_n, allocator=mempool)

        startTime = time.time()        
        for i in range(M):
            c = i*a + b + a + b + a + a + a
        calcTimeOpenCL1 = (time.time()-startTime)*1000

        startTime = time.time()        
        for i in range(M):
            c = i*a_n + b_n + a_n + b_n + a_n + a_n + a_n 
        calcTimeNumpy = (time.time()-startTime)*1000

        # Often, OpenCL is faster on the second attempt.
        startTime = time.time()        
        for i in range(M):
            c = i*a + b + a + b + a + a + a
        calcTimeOpenCL2 = (time.time()-startTime)*1000

        self.assertTrue(calcTimeOpenCL2 < calcTimeNumpy,msg="\nNumpy is faster than OpenCL: CL1 {0:.1f} ms NP {1:.1f} ms CL2 {2:.1f} ms".format(calcTimeOpenCL1, calcTimeNumpy, calcTimeOpenCL2))
        #print("\nCL1 {0:.1f} ms NP {1:.1f} ms".format(calcTimeOpenCL2, calcTimeNumpy))

    @unittest.skip("Skipping graphic tests")
    def test14PerformanceVsSize(self):
        """
        Performance with OpenCL is better but with very large arrays (2^20 or more)
        """

        # Set up basic OpenCl things
        queue = pycl.CommandQueue(self.context)
        allocator = pycl.tools.ImmediateAllocator(queue)
        mempool = pycl.tools.MemoryPool(allocator)

        N = 1
        M = 10
        P = 27
        nptimes = []
        cltimes = []
        for j in range(P):
            # Pre-allocate all arrays
            N = 1 << j
            a_n = np.random.rand(N).astype(np.float32)
            b_n = np.random.rand(N).astype(np.float32)

            # Pre-allocate opencl arrays, with MemoryPool to reuse memory
            a = pycl.array.to_device(queue=queue, ary=a_n, allocator=mempool)
            b = pycl.array.to_device(queue=queue, ary=b_n, allocator=mempool)

            startTime = time.time()        
            calcTimeOpenCL1 = []
            for i in range(M):
                c = i*a + b + a + b + a + a * a
            calcTimeOpenCL1.append((time.time()-startTime)*1000)
            cltimes.append(np.mean(calcTimeOpenCL1))

            startTime = time.time()        
            calcTimeOpenNP = []
            for i in range(M):
                c = i*a_n + b_n + a_n + b_n + a_n + a_n * a_n
            calcTimeOpenNP.append((time.time()-startTime)*1000)
            nptimes.append(np.mean(calcTimeOpenNP))

        plt.plot(range(P), np.divide(nptimes,cltimes), label="Speed OpenCL/numpy vs size")
        plt.plot(range(P), P*[1], label="Equal speed")
        plt.yscale('log')
        plt.xlabel("Size of array 2^x")
        plt.ylabel("Ratio of speed")
        plt.legend()
        plt.show()

    def test15_2x2Matrix_and_Vectors(self):
        """
        Here I am getting excited about the possiblities for RayTracing (2x2 matrices) 
        and I want to see it in action multiplying 2x2 matrices and vectors.

        """

        queue = pycl.CommandQueue(self.context)

        program_source = """
        kernel void product(global const float *mat, int M, 
                            global float *vec,
                            global float *res)
                      {
                      int i    = get_global_id(0); // the vector index
                      int j;                       // the matrix index

                      for (j = 0; j < M; j++) {
                          res[i + 2*j]     = vec[i];
                          res[i + 2*j + 1] = vec[i+1];

                          vec[i]     = mat[i+4*j]   * vec[i] + mat[i+4*j+1] * vec[i+1];
                          vec[i + 1] = mat[i+4*j+2] * vec[i] + mat[i+4*j+3] * vec[i+1];
                          }
                      }

        """
        program_source_floats = """
        kernel void product(global const float4 *mat, int M, 
                            global float2 *vec,
                            global float2 *res)
                      {
                      int i    = get_global_id(0); // the vector index
                      int j;                       // the matrix index
                      int N    = get_global_size(0);
                      float2 v = vec[i];
                      res[i] = v;
                      for (j = 0; j < M; j++) {
                          float4 m = mat[j];

                          v.x = m.x * v.x + m.y * v.y;
                          v.y = m.z * v.x + m.w * v.y;
                          res[i+N*(j+1)] = v;
                          }
                      }

        """
        program = pycl.Program(self.context, program_source_floats).build()
        knl = program.product  # Use this Kernel object for repeated calls


        startTime = time.time()        
        M = np.int32(40)    # M 2x2 matrices in path
        N = np.int32(2^24)  # N 2x1 rays to propagate
        # Pre-allocate opencl arrays, with MemoryPool to reuse memory
        matrix_n = np.random.rand(M,2,2).astype(np.float32)
        vector_n = np.random.rand(N,2).astype(np.float32)
        result_n = np.zeros((M+1,N,2)).astype(np.float32)

        matrix = pycl.array.to_device(queue=queue, ary=matrix_n)
        vector = pycl.array.to_device(queue=queue, ary=vector_n)
        result = pycl.array.to_device(queue=queue, ary=result_n)

        knl(queue, (N,), None, matrix.data, M, vector.data, result.data)

        # print("\n{0:0.1f} ms".format((time.time()-startTime)*1000))

    def test_16_call_random_value_with_same_seeds_buffer_should_give_same_results(self):
        """
        In an other project, we had a random number generator that was failing.  I am testing it here.

        """
        program_source =  """

            uint wangHash(uint seed){
                seed = (seed ^ 61) ^ (seed >> 16);
                seed *= 9;
                seed = seed ^ (seed >> 4);
                seed *= 0x27d4eb2d;
                seed = seed ^ (seed >> 15);
                return seed;
            }

            float getRandomFloatValue(__global unsigned int *seeds, int id){
                 float result = 0.0f;
                 while(result == 0.0f){
                     uint rnd_seed = wangHash(seeds[id]);
                     seeds[id] = rnd_seed;
                     result = (float)rnd_seed / (float)UINT_MAX;
                 }
                 return result;
            }

            // ----------------- TEST KERNELS -----------------

             __kernel void fillRandomFloatBuffer(__global unsigned int *seeds, __global float *randomNumbers){
                int id = get_global_id(0);
                randomNumbers[id] = getRandomFloatValue(seeds, id);
            }
            """
        program = pycl.Program(self.context, program_source).build()
        nWorkUnits = 32

        queue = pycl.CommandQueue(self.context)

        seedsBuffer1 = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)
        for i in range(nWorkUnits):
            seedsBuffer1[i] = i
        seedsBuffer2 = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)
        for i in range(nWorkUnits):
            seedsBuffer2[i] = i

        valueBuffer1 = clArray(cq=queue, shape=(nWorkUnits,), dtype=np.float32)
        valueBuffer2 = clArray(cq=queue, shape=(nWorkUnits,), dtype=np.float32)

        knl = program.fillRandomFloatBuffer

        knl(queue, (nWorkUnits,), None, seedsBuffer1.data, valueBuffer1.data)
        knl(queue, (nWorkUnits,), None, seedsBuffer2.data, valueBuffer2.data)

        self.assertTrue( all(valueBuffer1==valueBuffer2))

if __name__ == "__main__":
    unittest.main()
