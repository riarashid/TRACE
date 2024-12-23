# BO Solver - TRACE with n constraints
# Implements function minimization by default
# All decision variables are box-constrained in [0,1]

import scipy
import mopso_trace_ncons
import numpy as np
import torch
import gpytorch
import gaussianprocess
import math
import matplotlib.pyplot as plt
import copy as c
import pyDOE as pyd
import pandas as pd


def cboa(f_obj, f_constraint, dim, samples, solver='TRACE', init_sample_size=None, n_cons=1):
    if init_sample_size is None:
        init_sample_size = 2*dim + 4 # Initial sample size obtained from refernce "Multiproblem Surrogates IEEE TEVC paper"

    if samples <= init_sample_size:
        raise Exception('Available sample budget should be larger than initial sample size.')

    # Generate and evaluate initial solution samples
    #x_best, f_best, f_cons = generate_feasible_point(dim, f_obj, f_constraint)
    x_best = [0, 0]
    f_best = np.inf
    f_cons = 0
    #x = np.random.rand(init_sample_size, dim)
    x = generate_lhs(init_sample_size, dim)
    F = generate_init_samples(x, f_obj)
    F_c = generate_init_samples(x, f_constraint)
    constraint_count = []
    distance = []
    for i in range(0, samples-init_sample_size):
        count = 0
        y = np.array(F)
        # y = (y - y.mean()) / y.std()
        F_c = np.array(F_c)
        train_x = torch.from_numpy(x)
        train_y = torch.from_numpy(y)
        GP_info = train_GP(train_x, train_y)

        GP_info_cons = []
        if n_cons == 1:
            train_ycon = torch.from_numpy(F_c)
            GP_info_cons.append(train_GP(train_x, train_ycon))
        else:
            for k in range(n_cons):
                train_ycon = torch.from_numpy(F_c[:, k])
                GP_info_cons.append(train_GP(train_x, train_ycon))

        # call surrogate function optimizer and evaluate new generated point
        x_new = acquisition_function(dim, solver, y, GP_info, F_c, GP_info_cons, n_cons)
        x_new = np.array(x_new)
        f_new = f_obj(x_new)
        f_newc = f_constraint(x_new)
        x = np.concatenate((x, np.array([x_new])), 0)
        F.append(f_new)
        F_c = np.concatenate((F_c, np.array([f_newc])), 0)
        if n_cons == 1:
            if f_new > 0:
                count += 1
        else:
            if max(f_newc) > 0:
                count += 1
        # Update best solution found
        if (n_cons == 1 and (f_new < f_best) and f_newc <= 0) or (n_cons != 1 and (f_new < f_best) and max(f_newc) <= 0):
            f_best = f_new
            x_best = x_new
            f_cons = f_newc
        distance.append(f_best)
        constraint_count.append(count)
        print(i)
        print(f_new)
        print(f_newc)
        print(x_best)
        print(f_best)
        print(sum(constraint_count))
    plt_edisconv(distance, samples - init_sample_size)
    return x_best, f_best, f_cons, distance, constraint_count


def train_GP(train_x, train_y):
    likelihood, model = gaussianprocess.build_stgp(train_x, train_y)
    likelihood, model = gaussianprocess.train_stgp(train_x, train_y, likelihood, model)
    model.eval()
    model = model.double()
    likelihood.eval()
    GP_info = [model, likelihood]
    return GP_info


def generate_lhs(db, dim):
    X = pyd.lhs(dim, db, criterion='centermaximin')
    return X


def plt_edisconv(dis, points):
    xlen = []
    for i in range(points):
        xlen.append(i)
    fig = plt.figure()
    plt.plot(xlen, dis, color="green", linewidth=3)
    plt.title("BO_TRACE", fontsize=25)
    plt.ylabel('Fitness', fontsize=24)
    plt.xlabel('Evaluations', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig('Convplot_BO_TRACE.jpg')


def acquisition_function(dim, solver, y, gpinfo, yc, gpinfo_cons, n_cons):
    if solver == 'TRACE':
        x_new = mopso_trace_ncons.mopso(dim, f_lcb, f_ei, f_pi, f_cv1_trace, f_cv2_trace, y, gpinfo, yc, gpinfo_cons, n_cons, swarmsize=20, maxiter=100)
    else:
        raise Exception('No solver ' + solver + ' found.')
    return x_new


def f_lcb(x, GP_info):
    model = GP_info[0]
    likelihood = GP_info[1]
    test_x = torch.from_numpy(np.array([x]))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x.double()))
    return prediction.mean.detach().numpy()[0] - 0.3*prediction.stddev.detach().numpy()[0]


def f_pi(x, y, GP_info):
    model = GP_info[0]
    likelihood = GP_info[1]
    y_best = np.min(y)
    test_x = torch.from_numpy(np.array([x]))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x.double()))
    epsilon = 0.001
    predicted_mean = prediction.mean.detach().numpy()[0]
    predicted_stdev = prediction.stddev.detach().numpy()[0]
    lambda1 = ((y_best - predicted_mean - epsilon)/predicted_stdev)
    cdf = scipy.stats.norm.cdf(lambda1, loc=predicted_mean, scale=predicted_stdev)
    return -cdf


def f_ei(x, y, GP_info):
    model = GP_info[0]
    likelihood = GP_info[1]
    y_best = np.min(y)
    test_x = torch.from_numpy(np.array([x]))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x.double()))
    epsilon = 0.001
    lamda2 = ((y_best - prediction.mean.detach().numpy()[0] - epsilon) / prediction.stddev.detach().numpy()[0])
    pdf = scipy.stats.norm.pdf(lamda2, loc=prediction.mean.detach().numpy()[0], scale=prediction.stddev.detach().numpy()[0])
    cdf = scipy.stats.norm.cdf(lamda2, loc=prediction.mean.detach().numpy()[0],
                               scale=prediction.stddev.detach().numpy()[0])
    ei = prediction.stddev.detach().numpy()[0]*(lamda2*cdf + pdf)
    return -ei


def f_pf(x, GP_info_cons, n_cons):
    pf = np.ones(len(x))
    for k in range(n_cons):
        model = GP_info_cons[k][0]
        likelihood = GP_info_cons[k][1]
        test_xc = torch.from_numpy(np.array([x]))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(test_xc.double()))
        mean = prediction.mean.detach().numpy()[0]
        stdev = prediction.stddev.detach().numpy()[0]
        cdf = scipy.stats.norm.cdf(-mean/stdev, loc=mean, scale=stdev)
        pf *= cdf
    return -pf


def f_cv1_trace(x, GP_info_cons, n_cons):
    cv = []
    beta = 0.2
    for k in range(n_cons):
        model = GP_info_cons[k][0]
        likelihood = GP_info_cons[k][1]
        test_xc = torch.from_numpy(np.array([x]))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(test_xc.double()))
        mean = prediction.mean.detach().numpy()[0]
        stdev = prediction.stddev.detach().numpy()[0]
        cv.append(abs(mean - beta * stdev))
    cv = np.array(cv)
    cv_min = cv.min(axis=0)
    return cv_min


def f_cv2_trace(x, GP_info_cons, n_cons):
    cv = []
    beta = 0.2
    for k in range(n_cons):
        model = GP_info_cons[k][0]
        likelihood = GP_info_cons[k][1]
        test_xc = torch.from_numpy(np.array([x]))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(test_xc.double()))
        mean = prediction.mean.detach().numpy()[0]
        stdev = prediction.stddev.detach().numpy()[0]
        cv.append(mean - beta * stdev)
    cv = np.array(cv)
    cv_max = cv.max(axis=0)
    return cv_max


def f_cv1_mace(x, GP_info_cons,n_cons):
    cv = np.zeros(len(x))
    for k in range(n_cons):
        model = GP_info_cons[k][0]
        likelihood = GP_info_cons[k][1]
        test_xc = torch.from_numpy(np.array([x]))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(test_xc.double()))
        mean = prediction.mean.detach().numpy()[0]
        cv += np.maximum(0, mean)
    return cv


def f_cv2_mace(x, GP_info_cons,n_cons):
    cv = np.zeros(len(x))
    for k in range(n_cons):
        model = GP_info_cons[k][0]
        likelihood = GP_info_cons[k][1]
        test_xc = torch.from_numpy(np.array([x]))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(test_xc.double()))
        mean = prediction.mean.detach().numpy()[0]
        stdev = prediction.stddev.detach().numpy()[0]
        cv += np.maximum(0, mean/stdev)
    return cv


def generate_init_samples(x, f_obj):
    F = [f_obj(xi) for xi in x]
    return F


def generate_init_samples_constraints(x, f_obj):
    F_c = [f_obj(xi) for xi in x]
    return F_c


def generate_feasible_point(dim, f_obj, f_c):
    while True:
        new = np.random.rand(dim)
        F = f_obj(new)
        F_c = f_c(new)
        if max(F_c) <= 0:
            break
    return new, F, F_c




################# Testing code #########################################
if __name__ == "__main__":

    def sim1(x):
        x_local = c.deepcopy(x)
        x_local[0] = x_local[0] * 6
        x_local[1] = x_local[1] * 6
        fitness = math.cos(2 * x_local[0]) * math.cos(x_local[1]) + math.sin(x_local[0])
        return fitness


    def sim1_c(x):
        x_local = c.deepcopy(x)
        x_local[0] = x_local[0] * 6
        x_local[1] = x_local[1] * 6
        fitness_con = math.cos(x_local[0]) * math.cos(x_local[1]) - math.sin(x_local[0]) * math.sin(x_local[1]) - 0.5
        return fitness_con


    def testfun(x):
        x_local = c.deepcopy(x)
        fitness = (x_local[0] - 1) ** 2 + (x_local[1] - 0.5) ** 2
        return -fitness


    def testfun_c(x):
        x_local = c.deepcopy(x)
        a = -((x_local[1]) ** 7)
        c1 = ((x_local[0] - 3) ** 2 + ((x_local[1] + 2) ** 2)) * math.exp(a) - 12
        c2 = 10 * x_local[0] + x_local[1] - 7
        c3 = (x_local[0] - 0.5) ** 2 + (x_local[1] - 0.5) ** 2 - 0.2
        con = [c1, c2, c3]
        con = np.round(con, 3)
        return con

    dim = 2
    no_of_constraints = 3
    xbest_all = []
    fitness_all = []
    constraint_all = []
    for p in range(10):
        x_best, f_best, x, F, Constraintcount = cboa(testfun, testfun_c, dim, 200, solver='TRACE', init_sample_size=50, n_cons=no_of_constraints)
        xbest_all.append(x_best)
        fitness_all.append(F)
        constraint_all.append(Constraintcount)
        print('xbest', x_best)
        print('cv', np.sum(Constraintcount))

    # Saving data to excel
    xbest_array = np.array(xbest_all)
    fitness_array = np.array(fitness_all)
    constraint_array = np.array(constraint_all)
    excel_file_path = 'data_xbest.xlsx'
    df = pd.DataFrame(xbest_array)
    df.to_excel(excel_file_path, index=False)
    excel_file_path = 'data_constraint.xlsx'
    df = pd.DataFrame(constraint_array)
    df.to_excel(excel_file_path, index=False)
    excel_file_path = 'data_fitness.xlsx'
    df = pd.DataFrame(fitness_array)
    df.to_excel(excel_file_path, index=False)