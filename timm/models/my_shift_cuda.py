# the code is modified from https://github.com/d-li14/involution/blob/main/cls/mmcls/models/utils/involution_cuda.py
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template
import math

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    elif isinstance(t, torch.Tensor) and t.dtype == torch.float16:
        return 'half'


@cupy._util.memoize(for_each_device=True)
def load_my_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    # return cupy.RawKernel(code, kernel_name)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_my_shift_kernel = kernel_loop + '''
extern "C"
__global__ void my_shift_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};

    ${Dtype} value = 0;
    const int s1 = c % ${shift} - ${shift} / 2;
    const int s2 = (c / ${shift}) % ${shift} - ${shift} / 2;
    const int h_prime = h + s1;
    const int w_prime = w + s2;

    if ((h_prime >= 0 && h_prime < ${height}) && (w_prime >= 0 && w_prime < ${width})) {
        const int offset = ((n * ${channels} + c) * ${height} + h_prime) * ${width} + w_prime;
        value = bottom_data[offset];
    }

    top_data[index] = value;
  }
}
'''


_my_shift_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void my_shift_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    
    ${Dtype} value = 0;
    const int s1 = c % ${shift} - ${shift} / 2;
    const int s2 = (c / ${shift}) % ${shift} - ${shift} / 2;
    const int h_prime = h + s1;
    const int w_prime = w + s2;

    if ((h_prime >= 0 && h_prime < ${height}) && (w_prime >= 0 && w_prime < ${width})) {
        const int offset = ((n * ${channels} + c) * ${height} + h_prime) * ${width} + w_prime;
        value = top_diff[offset];
    }
    
    bottom_diff[index] = value;
  }
}
'''


class _my_shift(Function):
    @staticmethod
    def forward(ctx, input, shift):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = input.new(batch_size, channels, height, width)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_my_kernel('my_shift_forward_kernel', _my_shift_kernel, Dtype=Dtype(input), nthreads=n,
                num=batch_size, channels=channels, 
                height=height, width=width,
                shift=shift, group=int(math.ceil(channels/shift))
            )

            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input)
        ctx.shift = shift
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input = ctx.saved_tensors[0]
        shift = ctx.shift
        batch_size, channels, height, width = input.size()

        grad_input = None

        opt = dict(Dtype=Dtype(grad_output),
            num=batch_size, channels=channels,
            height=height, width=width,
            shift=shift, group=int(math.ceil(channels/shift))
        )

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_my_kernel('my_shift_backward_grad_input_kernel',
                                _my_shift_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, None, None
 

def _my_shift_cuda(input, shift):
    """ shift kernel
    """
    assert shift >= 3 and shift % 2 == 1

    if input.is_cuda:
        out = _my_shift.apply(input, shift)
    else:
        raise NotImplementedError
    return out


class MyShift(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        if self.kernel_size == 1:
            return x

        out = _my_shift_cuda(x, self.kernel_size)
        return out
    