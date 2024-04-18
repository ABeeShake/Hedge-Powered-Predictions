import numpy as np
import pandas as pd
import torch
import datetime

def get_data(path = "~/Simulations/t1d_trajectory.csv",dataset="t1d"):

    t1d = pd.read_csv(path)
    if dataset == "t1d":
        t1d.rename(columns={"Unnamed: 0":"Time"},inplace=True)

    return t1d

def get_t1d_vars(t1d, device, var = None):
    t = torch.Tensor(t1d.Time.to_numpy()).to(device)

    if var:
        X = torch.Tensor(t1d.iloc[:,var].to_numpy()).to(device)
    else:
        X = torch.Tensor(t1d.iloc[:,1:].to_numpy()).to(device)

    return t,X

def get_patientid(data, min_len = 300, max_len = 800):

    patient_lens = data.groupby("patientid")["egv"].count()
    possible_patients = patient_lens[(patient_lens >= min_len) & (patient_lens <= max_len)].index

    patientid = np.random.choice(possible_patients)

    print(patient_lens[patientid])

    return patientid

def get_patient_data(data, device, pid,covariates):

    patient_data = data[data["patientid"] == pid]
    X = torch.Tensor(patient_data[covariates].to_numpy()).to(device)
    y = torch.Tensor(patient_data["egv"].to_numpy()).to(device)

    return X,y

def get_cgm_vars(path,device,pid = None, covariates = ["age","weekday","month","tir_counts","below_55_counts","egv"]):

    data = pd.read_csv(path)
    data.loc[:,"age"] = data.loc[:,"date"].str.slice(0,4).astype(int) - data.loc[:,"dob"].str.slice(0,4).astype(int)
    data.loc[:,"weekday"] = data.loc[:,"date"].str.slice(0,10).apply(lambda x: (datetime.datetime.strptime(x,"%Y-%m-%d").weekday() + 1) % 7)
    data.loc[:,"month"] = data.loc[:,"date"].str.slice(5,7).astype(int)

    if pid:

        X,y = get_patient_data(data, device, pid, covariates)

    else :
        pid = get_patientid(data, min_len = 500, max_len = 1000)
        X,y = get_patient_data(data, device, pid, covariates)

    return X,y