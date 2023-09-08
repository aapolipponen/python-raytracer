import pyopencl as cl
import numpy as np
import os

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"

# Set up OpenCL environment
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Image dimensions
width, height = 1920, 1080

# Create an empty numpy array for the output image
output = np.empty((height, width, 4), dtype=np.uint8)  # 4 for RGBA

# Allocate memory on the GPU and copy data
output_gpu = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

# Load and compile the OpenCL program
with open("ray_tracing_kernel.cl", "r") as f:
    program_source = f.read()
program = cl.Program(context, program_source).build()

# Execute the kernel
global_work_size = (width, height)
program.ray_trace(queue, global_work_size, None, output_gpu, np.uint32(width), np.uint32(height))

# Copy the result back to the host
cl.enqueue_copy(queue, output, output_gpu)

# Save image using PIL
from PIL import Image
img = Image.fromarray(output, 'RGBA')
img.save('output.png')
