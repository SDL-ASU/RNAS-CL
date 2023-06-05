from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn.functional as F

def pgd_whitebox(model, X, y, random_start=True,
                      epsilon=8/255, num_steps=10, step_size=0.003):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        
        return X_pgd
        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()
        return acc.item(), acc_pgd.item(), loss.item(), stable.item(), X_pgd


def cw_whitebox(model, X, y, random_start=True, epsilon=8/255, num_steps=20, step_size=0.003):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)

        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                correct_logit = torch.sum(torch.gather(model(X_pgd), 1, (y.unsqueeze(1)).long()).squeeze())
                tmp1 = torch.argsort(model(X_pgd), dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                wrong_logit = torch.sum(torch.gather(model(X_pgd), 1, (new_y.unsqueeze(1)).long()).squeeze())
                loss = - F.relu(correct_logit-wrong_logit)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        return X_pgd

'''
    def _cw_whitebox(self, model, X, y, random_start=True,
                     epsilon=0.031, num_steps=20, step_size=0.003):
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)

        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                correct_logit = torch.sum(torch.gather(model(X_pgd), 1, (y.unsqueeze(1)).long()).squeeze())
                tmp1 = torch.argsort(model(X_pgd), dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                wrong_logit = torch.sum(torch.gather(model(X_pgd), 1, (new_y.unsqueeze(1)).long()).squeeze())
                loss = - F.relu(correct_logit-wrong_logit)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()

        return acc.item(), acc_pgd.item(), loss.item(), stable.item(), X_pgd

'''

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


'''
def pgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
):

    x_pgd = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([x_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_pgd), y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, clip_min, clip_max), requires_grad=True)

    return x_pgd

'''
