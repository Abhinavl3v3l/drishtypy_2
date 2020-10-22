import torch
from torchviz import make_dot
from torch.autograd import Variable

def graphical_summary(model, use_cuda=True, save=True):
    random_input = torch.randn(1, 3, 32, 32).cuda() if use_cuda else torch.randn(1, 3, 32, 32)
    model.eval()
    mod = model(Variable(random_input))
    dot_graph = make_dot(mod)
    if save:
        dot_graph.format = 'svg'
        dot_graph.render(f'model_architecture')
    return dot_graph