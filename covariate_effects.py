
if __name__ == "__main__":

    import Hedge.models as m
    import Hedge.data as d
    import Hedge.simulate as sim
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sys
    import csv

    path, num_samples = sys.argv[1:]

    data = pd.read_csv(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pid = d.get_patientid(data, min_len = 500, max_len = 1000)

    X_cov,y_cov = d.get_cgm_vars(path, device, pid = pid)
    X_ncov, y_ncov = d.get_cgm_vars(path, device, pid = pid, covariates=["tir_counts","below_55_counts","egv"])

    dim_cov = X_cov.shape[-1]
    dim_ncov = X_ncov.shape[-1]

    start = 20

    for i in range(int(num_samples)):

        end = len(X_cov)

        eta = np.sqrt(np.log(3)/(end - start))

        forecasts_cov = sim.run_sim(start, end, device, X_cov, y = None, save = False)
        forecasts_ncov = sim.run_sim(start, end, device, X_ncov, y = None, save = False)

        #ftl_cov, losses_cov, ftl_losses_cov, leader_cov = sim.full_loss_forecasts(forecasts_cov, y_cov, start = start, end = end, device = device, eta = eta)
        ftl2_cov, losses2_cov, leader2_cov = sim.leader_loss_forecasts(forecasts_cov, y_cov, start = start, end = end, device = device, eta = eta)

        #ftl_ncov, losses_ncov, ftl_losses_ncov, leader_ncov = sim.full_loss_forecasts(forecasts_ncov, y_ncov, start = start, end = end, device = device, eta = eta)
        ftl2_ncov, losses2_ncov, leader2_ncov = sim.leader_loss_forecasts(forecasts_ncov, y_ncov, start = start, end = end, device = device, eta = eta)

        ftl_losses2_cov = losses2_cov.flatten()[losses2_cov.flatten().nonzero()]
        ftl_losses2_ncov = losses2_ncov.flatten()[losses2_ncov.flatten().nonzero()]

        with open("./losses/leader_cov_losses.csv", "a") as f:
    
            w = csv.writer(f)
            w.writerows(ftl_losses2_cov.tolist())

        with open("./losses/leader_nocov_losses.csv", "a") as f:
    
            w = csv.writer(f)
            w.writerows(ftl_losses2_ncov.tolist())

        pid = d.get_patientid(data, min_len = 500, max_len = 1000)

        X_cov,y_cov = d.get_cgm_vars(path, device, pid = pid)
        X_ncov, y_ncov = d.get_cgm_vars(path, device, pid = pid, covariates=["tir_counts","below_55_counts","egv"])