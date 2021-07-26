import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch import Tensor
from cupy_layers.utils import Dtype, Stream, load_kernel

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

_aggregation_zeropad_dilate_forward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_dilate_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, const ${Dtype}* dilation_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${weight_heads} / ${input_channels} / ${top_height} / ${top_width};
    const int head = (index / ${top_width} / ${top_height} / ${input_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${input_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int c_w = c % ${weight_channels};
    const int dilation_h = dilation_data[c_w];
    const int dilation_w = dilation_h;
    const int pad_h = dilation_h;  //const int pad_h = ((${stride_h} - 1) + dilation_h * (${kernel_h} - 1)) / 2;
    const int pad_w = dilation_w;  //const int pad_w = ((${stride_w} - 1) + dilation_w * (${kernel_w} - 1)) / 2;

    ${Dtype} value = 0;
    const int _kernel_h = 3;
    const int _kernel_w = 3;
    //for (int kh = 0; kh < ${kernel_h}; ++kh) {
      //for (int kw = 0; kw < ${kernel_w}; ++kw) {
    #pragma unroll
    for (int kh = 0; kh < _kernel_h; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < _kernel_w; ++kw) {
        const int h_in = -pad_h + h + kh * dilation_h;  //const int h_in = -pad_h + h * ${stride_h} + kh * dilation_h;
        const int w_in = -pad_w + w + kw * dilation_w;  //const int w_in = -pad_w + w * ${stride_w} + kw * dilation_w;
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
          const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset_bottom];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_aggregation_zeropad_dilate_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_dilate_input_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, const ${Dtype}* dilation_data, ${Dtype}* bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};

    const int c_w = c % ${weight_channels};
    const int dilation_h = dilation_data[c_w];
    const int dilation_w = dilation_h;
    const int pad_h = dilation_h;  //const int pad_h = ((${stride_h} - 1) + dilation_h * (${kernel_h} - 1)) / 2;
    const int pad_w = dilation_w;  //const int pad_w = ((${stride_w} - 1) + dilation_w * (${kernel_w} - 1)) / 2;

    ${Dtype} value = 0;
    const int _kernel_h = 3;
    const int _kernel_w = 3;

    for (int head = 0; head < ${weight_heads}; ++head) {
        //for (int kh = 0; kh < ${kernel_h}; ++kh) {
          //for (int kw = 0; kw < ${kernel_w}; ++kw) {
        #pragma unroll
        for (int kh = 0; kh < _kernel_h; ++kh) {
          #pragma unroll
          for (int kw = 0; kw < _kernel_w; ++kw) {
            const int h_out_s = h + pad_h - kh * dilation_h;
            const int w_out_s = w + pad_w - kw * dilation_w;
            //if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
              const int h_out = h_out_s;  //const int h_out = h_out_s / ${stride_h};
              const int w_out = w_out_s;  //const int w_out = w_out_s / ${stride_w};
              if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
                const int offset_top = (((n * ${weight_heads} + head) * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
                const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
                value += weight_data[offset_weight] * top_diff[offset_top];
              }
            //}
          }
        }
    }
    bottom_diff[index] = value;
  }
}
'''

_aggregation_zeropad_dilate_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_dilate_weight_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, const ${Dtype}* dilation_data, ${Dtype}* weight_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${weight_heads} / ${weight_channels} / ${top_height} / ${top_width};
    const int head = (index / ${top_width} / ${top_height} / ${weight_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${weight_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};

    const int dilation_h = dilation_data[c];
    const int dilation_w = dilation_h;
    const int pad_h = dilation_h;  //const int pad_h = ((${stride_h} - 1) + dilation_h * (${kernel_h} - 1)) / 2;
    const int pad_w = dilation_w;  //const int pad_w = ((${stride_w} - 1) + dilation_w * (${kernel_w} - 1)) / 2;

    const int _kernel_h = 3;
    const int _kernel_w = 3;

    //for (int kh = 0; kh < ${kernel_h}; ++kh) {
    //  for (int kw = 0; kw < ${kernel_w}; ++kw) {
    #pragma unroll
    for (int kh = 0; kh < _kernel_h; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < _kernel_w; ++kw) {
        const int h_in = -pad_h + h + kh * dilation_h;  //const int h_in = -pad_h + h * ${stride_h} + kh * dilation_h;
        const int w_in = -pad_w + w + kw * dilation_w;  //const int w_in = -pad_w + w * ${stride_w} + kw * dilation_w;
        const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
        ${Dtype} value = 0;
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
            const int offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_top = (((n * ${weight_heads} + head) * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
            value += bottom_data[offset_bottom] * top_diff[offset_top];
          }
        }
        weight_diff[offset_weight] = value;
      }
    }
  }
}
'''

class AggregationZeropadDilate(Function):
    @staticmethod
    def forward(ctx, input, weight, dilation, kernel_size, stride):
        kernel_size, stride = _pair(kernel_size), _pair(stride)
        ctx.kernel_size, ctx.stride = kernel_size, stride
        assert input.dim() == 4 and input.is_cuda and weight.is_cuda and dilation.is_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_heads, weight_channels, weight_kernels, weight_height, weight_width = weight.size()
        output_height = input_height
        output_width = input_width
        assert output_height * output_width == weight_height * weight_width
        output = input.new(batch_size, weight_heads * input_channels, output_height, output_width)
        n = output.numel()
        if not input.is_contiguous():
            input = input.detach().clone()
        if not weight.is_contiguous():
            weight = weight.detach().clone()
        if not dilation.is_contiguous():
            dilation = dilation.detach().clone()

        with torch.cuda.device_of(input):
            f = load_kernel('aggregation_zeropad_dilate_forward_kernel', _aggregation_zeropad_dilate_forward_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, input_channels=input_channels, 
                            weight_heads=weight_heads, weight_channels=weight_channels,
                            bottom_height=input_height, bottom_width=input_width,
                            top_height=output_height, top_width=output_width,
                            kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                            stride_h=stride[0], stride_w=stride[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), dilation.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input, weight, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, stride = ctx.kernel_size, ctx.stride
        input, weight, dilation = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_heads, weight_channels, weight_kernels, weight_height, weight_width = weight.size()
        output_height, output_width = grad_output.size()[2:]
        grad_input, grad_weight = None, None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, input_channels=input_channels, 
                   weight_heads=weight_heads, weight_channels=weight_channels,
                   bottom_height=input_height, bottom_width=input_width,
                   top_height=output_height, top_width=output_width,
                   kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                   stride_h=stride[0], stride_w=stride[1])
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_dilate_input_backward_kernel', _aggregation_zeropad_dilate_input_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), dilation.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())
                n = grad_weight.numel() // weight.shape[3]
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_dilate_weight_backward_kernel', _aggregation_zeropad_dilate_weight_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), dilation.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, grad_weight, None, None, None


def aggregation_zeropad_dilate(input, weight, dilation, kernel_size=3, stride=1):
    assert (input.shape[0] == weight.shape[0]) and (input.shape[1] % weight.shape[2] == 0) and (dilation.shape[0] == weight.shape[2])
    if input.is_cuda:
        out = AggregationZeropadDilate.apply(input, weight, dilation, kernel_size, stride)
    else:
        #raise NotImplementedError
        out = AggregationZeropadDilate.apply(input.cuda(), weight.cuda(), dilation.cuda(), kernel_size, stride)
        torch.cuda.synchronize()
        out = out.cpu()
    return out

class LocalConvolutionDilate(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super(LocalConvolutionDilate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        assert kernel_size == 3

    def forward(self, input: Tensor, weight: Tensor, dilation: Tensor):
        out = aggregation_zeropad_dilate(
            input, 
            weight, 
            dilation,
            kernel_size=self.kernel_size, 
            stride=self.stride)
        return out

def test_aggregation_zeropad_dilate():
    kernel_size, stride = 3, 1
    head_num = 2
    n, c_x, c_w, in_height, in_width = 2, 8, 4, 7, 7
    out_height = in_height
    out_width = in_width
    dilation_arr = [1, 1, 2, 4]
    split_arr = [2, 1, 1]
    padding = [d * (kernel_size - 1) // 2 for d in dilation_arr]
    dilation = torch.tensor(dilation_arr).double().cuda()

    x = torch.randn(n, c_x, in_height, in_width, requires_grad=True).double().cuda()
    w = torch.randn(n, head_num, c_w, pow(kernel_size, 2), out_height, out_width, requires_grad=True).double().cuda()
    w1, w2, w3 = torch.split(w, split_arr, dim=2)

    _x = x.view(n, c_x//len(dilation_arr), len(dilation_arr), in_height, in_width)
    x1, x2, x3 = torch.split(_x, split_arr, dim=2)
    x1 = x1.reshape(n, -1, in_height, in_width)
    x2 = x2.reshape(n, -1, in_height, in_width)
    x3 = x3.reshape(n, -1, in_height, in_width)

    y1 = aggregation_zeropad_dilate(x, w, dilation, kernel_size=kernel_size, stride=stride)
    unfold_j1 = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation_arr[0], padding=padding[0], stride=stride)
    unfold_j2 = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation_arr[2], padding=padding[2], stride=stride)
    unfold_j3 = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation_arr[3], padding=padding[3], stride=stride)
    x11 = unfold_j1(x1).view(n, x1.shape[1]//split_arr[0], split_arr[0], pow(kernel_size, 2), out_height, out_width)
    x22 = unfold_j2(x2).view(n, x2.shape[1]//split_arr[1], split_arr[1], pow(kernel_size, 2), out_height, out_width)
    x33 = unfold_j3(x3).view(n, x3.shape[1]//split_arr[2], split_arr[2], pow(kernel_size, 2), out_height, out_width)

    y11 = (w1.unsqueeze(2) * x11.unsqueeze(1)).sum(-3).view(n, head_num * x1.shape[1], out_height, out_width)
    y22 = (w2.unsqueeze(2) * x22.unsqueeze(1)).sum(-3).view(n, head_num * x2.shape[1], out_height, out_width)
    y33 = (w3.unsqueeze(2) * x33.unsqueeze(1)).sum(-3).view(n, head_num * x3.shape[1], out_height, out_width)

    y11 = y11.view(n, -1, split_arr[0], out_height, out_width)
    y22 = y22.view(n, -1, split_arr[1], out_height, out_width)
    y33 = y33.view(n, -1, split_arr[2], out_height, out_width)
    y2 = torch.cat([y11, y22, y33], dim=2)
    y2 = y2.view(n, -1, out_height, out_width)
    assert (y1 - y2).abs().max() < 1e-9

    gx1 = torch.autograd.grad(y1.mean(), x, retain_graph=True)[0]
    gx2 = torch.autograd.grad(y2.mean(), x, retain_graph=True)[0]
    assert (gx1 - gx2).abs().max() < 1e-9

    gw1 = torch.autograd.grad(y1.mean(), w, retain_graph=True)[0]
    gw2 = torch.autograd.grad(y2.mean(), w, retain_graph=True)[0]
    assert (gw1 - gw2).abs().max() < 1e-9

    from functools import partial
    assert torch.autograd.gradcheck(partial(aggregation_zeropad_dilate, dilation=dilation, kernel_size=kernel_size, stride=stride), (x, w))
    print('test case passed')


if __name__ == '__main__':
    test_aggregation_zeropad_dilate()