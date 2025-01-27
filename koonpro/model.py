from torch.distributions import Normal
from utils import *

class Koonpro(torch.nn.Module):
    def __init__(self, args, enc_width, dec_width, aux_width, dmd_width):
        print('Now use Koonpro')
        super().__init__()
        self.size = args.dims
        self.window = args.window
        self.encoder = MLN_MLP(enc_width)
        self.decoder = MLN_MLP(dec_width)
        self.the_mu = MLN_MLP(aux_width)
        self.the_sigma = MLN_MLP(aux_width)
        self.dropout = torch.nn.Dropout(args.dropout)
        self.var_weight = args.var_weight
        self.var_weight_decoder = args.var_weight_decoder

        self.dmd_to_hidden = MLN_MLP(dmd_width)
        self.r_to_hidden = torch.nn.Linear(dmd_width[-1], 2*dmd_width[-1])
        self.hidden_to_mu = torch.nn.Linear(2*dmd_width[-1], dmd_width[-1])
        self.hidden_to_sigma = torch.nn.Linear(2*dmd_width[-1], dmd_width[-1])

    def q_cal(self, X, Y):
        x, y = X[:, :, :, 0], Y[:, :, :, 0]
        x, y = torch.transpose(x, 1, 2), torch.transpose(y, 1, 2)
        A = torch.matmul(torch.linalg.pinv(x), y)
        A = A.view(A.shape[0], A.shape[1]*A.shape[2])
        r = self.dmd_to_hidden(A)
        r = torch.mean(r, 0)
        r = r.view(1, -1)
        h = self.r_to_hidden(r)

        mu = self.hidden_to_mu(h)
        sigma = self.hidden_to_sigma(h)
        sigma = torch.sigmoid(sigma)
        sigma = 0.1 + 0.9 * sigma
        mu = torch.transpose(mu, 1, 0)
        sigma = torch.transpose(sigma, 1, 0)
        return mu, sigma

    def q_predict(self, X, Y):
        x, y = X[:, :, :, 0], Y[:, :, :, 0]
        x, y = torch.transpose(x, 1, 2), torch.transpose(y, 1, 2)
        A = torch.matmul(torch.linalg.pinv(x), y)
        A = A.view(A.shape[0], A.shape[1]*A.shape[2])
        r = self.dmd_to_hidden(A)
        h = self.r_to_hidden(r)

        mu = self.hidden_to_mu(h)
        sigma = self.hidden_to_sigma(h)
        sigma = torch.sigmoid(sigma)
        sigma = 0.1 + 0.9 * sigma
        mu = torch.transpose(mu, 1, 0)
        sigma = torch.transpose(sigma, 1, 0)
        return mu, sigma

    def forward(self, data, time_shifts, linear_shifts, delta_t, q):
        y_list = []
        x_list_mu = []
        x_list_sigma = []
        p_list = []
        q = torch.squeeze(q)
        self.q = q.repeat(data.shape[0], 1)
        self.data = torch.transpose(data, 0, 1)
        self.data = self.data.view(self.data.shape[0], self.data.shape[1], -1)

        y_advanced = self.dropout(self.encoder(self.data[0]))
        for j in range(1, linear_shifts+1):
            x = self.data[j]
            y_temp = self.dropout(self.encoder(x))
            y_list.append(y_temp)
        for j in range(time_shifts):
            f = torch.cat((y_advanced, self.q), 1)
            the = f
            the_mu = self.the_mu(the)
            the_sigma = self.the_sigma(the)
            y_advanced_mu = varying_multiply(y_advanced, the_mu, delta_t)
            y_advanced_sigma = varying_multiply(y_advanced, the_sigma, delta_t)
            y_advanced_sigma = self.var_weight * 0.1 + self.var_weight * 0.9 * torch.sigmoid(y_advanced_sigma)
            y_advanced = Normal(y_advanced_mu, y_advanced_sigma).rsample()

            x_pred_mu = self.decoder(y_advanced_mu)
            x_pred_sigma = self.var_weight_decoder * 0.1 + self.var_weight_decoder * 0.9 * torch.sigmoid(self.decoder(y_advanced_sigma))
            x_pred_mu, x_pred_sigma = x_pred_mu.view(x_pred_mu.shape[0], self.size, self.window), x_pred_sigma.view(
                x_pred_sigma.shape[0], self.size, self.window)
            x_list_mu.append(x_pred_mu)  # x_list contains distribution of prediction
            x_list_sigma.append(x_pred_sigma)
            p_list.append(y_advanced)
        x_list_mu = torch.stack(x_list_mu)
        x_list_sigma = torch.stack(x_list_sigma)
        x_list_mu, x_list_sigma = torch.transpose(x_list_mu, 0, 1), torch.transpose(x_list_sigma, 0, 1)
        p_list, y_list = torch.stack(p_list), torch.stack(y_list)
        p_list = p_list[:-1, :, :]
        return data[:, 1:, :, :], x_list_mu, x_list_sigma, p_list, y_list

    def predict(self, data, time_shifts, delta_t, q):
        x_list_mu = []
        x_list_sigma = []
        self.q = torch.squeeze(q)
        self.data = torch.transpose(data, 0, 1)
        self.data = self.data.view(self.data.shape[0], self.data.shape[1], -1)
        y_advanced = self.dropout(self.encoder(self.data[0]))

        for j in range(time_shifts):
            f = torch.cat((y_advanced, self.q), 1)
            the = f
            the_mu = self.the_mu(the)
            the_sigma = self.the_sigma(the)
            y_advanced_mu = varying_multiply(y_advanced, the_mu, delta_t)
            y_advanced_sigma = varying_multiply(y_advanced, the_sigma, delta_t)
            y_advanced_sigma = self.var_weight * 0.1 + self.var_weight * 0.9 * torch.sigmoid(y_advanced_sigma)
            y_advanced = Normal(y_advanced_mu, y_advanced_sigma).rsample()

            x_pred_mu = self.decoder(y_advanced_mu)
            x_pred_sigma = 0.1 + 0.9 * torch.sigmoid(self.decoder(y_advanced_sigma))
            x_pred_mu, x_pred_sigma = x_pred_mu.view(x_pred_mu.shape[0], self.size, self.window), x_pred_sigma.view(
                x_pred_sigma.shape[0], self.size, self.window)
            x_list_mu.append(x_pred_mu)  # x_list contains distribution of prediction
            x_list_sigma.append(x_pred_sigma)
        x_list_mu = torch.stack(x_list_mu)
        x_list_sigma = torch.stack(x_list_sigma)
        x_list_mu, x_list_sigma = torch.transpose(x_list_mu, 0, 1), torch.transpose(x_list_sigma, 0, 1)
        return data[1:, :, :, :], x_list_mu, x_list_sigma