'''import example_bind
import torch

input = torch.ones(16, 1024, 32, 64, device='cuda')
output = torch.zeros_like(input)
example_bind.copy_kernel(input, output)
print(output.mean(), '\n')

output = torch.zeros_like(input)
example_bind.wrapped_copy_kernel(input, output)
print(output.mean())
'''
# rewrite
import example_bind
import torch
input = torch.ones(16, 1024, 32, 64, device='cuda')
output = torch.zeros_like(input)
example_bind.copy_kernel(input, output)
print(f"using 1s {output.mean()=}\n")

output = torch.zeros_like(input)
example_bind.wrapped_copy_kernel(input, output)
print(f"second output {output.mean()}")
