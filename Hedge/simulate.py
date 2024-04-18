import torch
import Hedge.models as m
import matplotlib.pyplot as plt
import numpy as np

def run_sim(start, end, device, X, y = None, save = False):

    dim = X.shape[-1]
    forecasts = torch.zeros(dim,end-start,3).to(device)

    node = m.neural_ode(dim,device).to(device)
    rnn = m.RNN(dim).to(device)
    transformer = m.TransformerModel(input_dim = dim).to(device)

    for i,k in enumerate(range(start,end)):

        print(k)

        future = X[k].reshape((1,-1))
        x_train = X[start-1:k]
        y_train = y[start-1:k] if torch.is_tensor(y) else None


        if i < 5:

            continue        

        t = torch.linspace(0,1,i)

        if k < 300:

            if k%20 == 0:

                m.train_lstm(rnn,x_train,y_train)

            if k%100 == 0:

                m.train_node(t, node, X = x_train,y = y_train)
                m.train_transformer(transformer, x_train,y_train, epochs = 20)


            forecasts[:,i,0] = m.node_forecast(node, future)
            forecasts[:,i,1] = rnn(future)
            forecasts[:,i,2] = transformer(future)

        else:

            if k%100 == 0:

                m.train_node(t, node, X = x_train,y = y_train)
                m.train_transformer(transformer, x_train,y_train, epochs = 20)

            if k%30 == 0 and (i+30 < end - start + 1):

                m.train_lstm(rnn,x_train,y_train)

                forecasts[:,i:i+30,0] = m.node_forecast(node, X[k:k+30]).T
                forecasts[:,i:i+30,1] = rnn(X[k:k+30]).T
                forecasts[:,i:i+30,2] = transformer(X[k:k+30]).T

            elif k%30 == 0:

                m.train_lstm(rnn,x_train,y_train)

                forecasts[:,i:,0] = m.node_forecast(node, X[k:end+1]).T
                forecasts[:,i:,1] = rnn(X[k:end+1]).T
                forecasts[:,i:,2] = transformer(X[k:end+1]).T


    if save:
        
        torch.save(node.state_dict(), "./models/neural_ode.pt")
        torch.save(rnn.state_dict(), "./models/lstm.pt")
        torch.save(transformer.state_dict(), "./models/transformer.pt")

    return forecasts

def plot_losses(losses):

    losses = losses.cpu().detach()

    plt.plot(losses.cumsum(axis=1)[2,:,0], label = "Neural ODE")
    plt.plot(losses.cumsum(axis=1)[2,:,1], label = "RNN")
    plt.plot(losses.cumsum(axis=1)[2,:,2], label = "Transformer")
    #plt.plot(losses.cumsum(axis=1)[2,:,3], label = "SINDy")#
    plt.legend()
    plt.title("Cumulative Losses")
    plt.show()
    return None

def plot_forecasts(forecasts, y, ftl, patientid, start, end):

    forecasts = forecasts.cpu().detach()
    y = y.cpu().detach()
    ftl = ftl.cpu().detach()

    plt.plot(y[start:end+1], c = "black", label = "True Value")
    plt.plot(forecasts[-1,:end-start+1,0], c = "lightblue", label = "Neural ODE")
    plt.plot(forecasts[-1,:end-start+1,1], c = "orange", label = "RNN")
    plt.plot(forecasts[-1,:end-start+1,2], c = "blue", label = "Transformer")
    plt.plot(ftl[-1,:end-start+1], "--", c = "red", label = "Hedge")
    plt.title(f"Estimate Glucose Forecasts for Patient {patientid}")
    plt.legend()
    plt.show()
    return None

def full_loss_forecasts(forecasts, y, device, start, end, eta):

    dim = forecasts.shape[0]
    n_models = forecasts.shape[-1]

    losses = torch.zeros((dim,end-start,n_models)).to(device)

    for i in range(n_models):

        losses[:,:,i] = ((forecasts[:,:,i] - y[start:end].T)**2)

    losses = losses / losses.max()

    dim = forecasts.shape[0]

    weights = torch.exp(-eta * losses.cumsum(axis=1))
    leader = weights.argmax(axis=2)

    ftl = torch.zeros(forecasts.shape[:-1])
    ftl_losses = torch.zeros(forecasts.shape[:-1])

    for v in range(dim):

        ftl[v,:] = torch.Tensor([forecasts[v,i,j] for i,j in zip(range(end),leader[v,:])])
        ftl_losses[v,:] = torch.Tensor([losses[v,i,j] for i,j in zip(range(end),leader[v,:])])

    return ftl, losses, ftl_losses, leader

def leader_loss_forecasts(forecasts, y, device, start, end, eta):

    dim = forecasts.shape[0]
    n_models = forecasts.shape[-1]

    losses = torch.ones((dim,end-start,n_models)).to(device)
    leaders = torch.zeros(end-start)

    ftl = torch.zeros(forecasts.shape[:-1])

    for v in range(dim):

        weights = torch.ones(n_models)
        p = weights / weights.sum()

        for i,k in enumerate(range(start, end)):

            leader = torch.multinomial(p, num_samples=1)
            leaders[i] = leader
            ftl[v,i] = forecasts[v,i,leader]
            ell = (ftl[v,i] - y[k])**2
            losses[v,i,leader] = ell
            losses[v,i,:] = losses[v,i,:] / losses[v,i,:].sum()

            weights = weights * torch.exp(-eta * losses[v,i,:])
            p = weights / weights.sum()

    return ftl, losses, leaders