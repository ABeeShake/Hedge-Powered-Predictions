
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

    pids = data["patientid"].unique()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = 20

    for i in range(int(num_samples)):

        X,y = d.get_cgm_vars(path, device, pid = pids[i])

        end = len(X)

        eta = np.sqrt(np.log(3)/(end - start))

        forecastsi = sim.run_sim(start, end, device, X, y = None, save = False)

        ftli, lossesi, ftl_lossesi, leaderi = sim.full_loss_forecasts(forecastsi, y, start = start, end = end, device = device, eta = eta)
        ftl2i, losses2i, leader2i = sim.leader_loss_forecasts(forecastsi, y, start = start, end = end, device = device, eta = eta)

        ftl_losses2i = losses2i.flatten()[losses2i.flatten().nonzero()]

        with open("./losses/full_losses.csv", "a") as f:
    
            w = csv.writer(f)
            w.writerows(ftl_lossesi.T.tolist())

        with open("./losses/leader_losses.csv", "a") as f:
    
            w = csv.writer(f)
            w.writerows(ftl_losses2i.T.tolist())