# overview 

notes about how pytorch implements autograd. 

[pytroch's doc on autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html)
[知乎解释 backward](https://zhuanlan.zhihu.com/p/83172023)
[good discussion on threads](https://zh.d2l.ai/chapter_preliminaries/autograd.html)

## AD with control flow 
in pytorch's framework. for control flow functions, AD is implemented as derivative based on execution path. Even if function is not differentiable around the condition's boundary. pytorch just reports back the gradient of executed path. 

## example 1
```python 
def f(a):
    b = a * 2
    if b >= 0:
        b = b * 2
    return b


# a = torch.randn(size=(), requires_grad=True)
a = torch.zeros(size=(), requires_grad=True)
z = f(a)
z.backward()
a.grad

# return 4
```

## example 2 
```python 
def f(a):
    b = a * 2
    if b > 0: # change boundary to not activate when b == 0
        b = b * 2
    return b


# a = torch.randn(size=(), requires_grad=True)
a = torch.zeros(size=(), requires_grad=True)
z = f(a)
z.backward()
a.grad
# return 2
# return 2
```
