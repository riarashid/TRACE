import torch
import gpytorch
import stgp


def build_stgp(train_x, train_y):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = stgp.ExactGPModel(train_x, train_y, likelihood)

    return likelihood, model


def train_stgp(train_x, train_y, likelihood, model):

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return likelihood, model


################# Testing code #########################################
if __name__ == "__main__":
    print('Done')