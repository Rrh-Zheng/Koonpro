from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch.nn as nn
from torch import optim
from utils import *
import time
from model import Koonpro
from data.dataloader_gen import data_provider
from exp.exp_basic import Exp_Basic
from metric import get_metric, plot_interval


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        base = self.args.window * self.args.dims
        self.enc_width = []
        self.dec_width = []
        self.aux_width = []
        self.dmd_width = []
        self.enc_width.append(base)
        self.enc_width = self.enc_width + self.args.enc_width
        self.enc_width.append(base * 2)

        self.dec_width.append(base * 2)
        self.dec_width = self.dec_width + self.args.enc_width
        self.dec_width.append(base)

        self.dmd_width.append(self.args.time_shifts * self.args.time_shifts)
        self.dmd_width = self.dmd_width + self.args.dmd_width

        self.aux_width.append(2 * base + self.dmd_width[-1])
        self.aux_width = self.aux_width + self.args.aux_width
        self.aux_width.append(base)

        model = Koonpro(self.args, self.enc_width, self.dec_width, self.aux_width, self.dmd_width)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        context, data_org, dataloader = data_provider(self.args, flag)
        return context, data_org, dataloader

    def _get_data_snr(self, flag):
        context, data_org, dataloader = data_provider(self.args, flag)
        return context, data_org, dataloader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, loader, q_context):
        epoch_loss_pred_val = 0
        epoch_loss_lin_val = 0
        epoch_kl_val = 0
        self.model.eval()
        loader_size = len(loader)
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.float()
                data = data.to(self.device)
                X, Y = data[:, 0:self.args.time_shifts, :, :], data[:, 1:self.args.time_shifts + 1, :, :]

                X, Y = X.to(self.device), Y.to(self.device)
                mu_target, sigma_target = self.model.q_cal(X, Y)
                q_target_test = Normal(mu_target, sigma_target)
                q_target_resample_test = q_target_test.rsample()
                q_target_resample_test = q_target_resample_test.repeat(torch.cuda.device_count(), 1)
                y, x_list_mu, x_list_sigma, p_list, y_list = self.model(data, self.args.time_shifts,
                                                                        self.args.time_shifts - 1, self.args.delta_t,
                                                                        q_target_resample_test)

                if self.args.get_back:
                    x_list_mu, x_list_sigma, y = get_back(x_list_mu, x_list_sigma, y)
                    x_prob = Normal(x_list_mu, x_list_sigma)
                    x_sample = x_prob.rsample()
                else:
                    x_prob = Normal(x_list_mu, x_list_sigma)
                    x_sample = x_prob.rsample()

                loss_pred = 0
                if self.args.use_prob:
                    loss_pred = loss_pred - x_prob.log_prob(y).mean()
                else:
                    loss_pred = nn.MSELoss()(x_sample, y)

                loss_lin = nn.MSELoss()(p_list, y_list)
                kl = kl_divergence(q_context, q_target_test).mean()

                epoch_loss_pred_val += loss_pred.item()
                epoch_loss_lin_val += loss_lin.item()
                epoch_kl_val += kl.item()

            return epoch_loss_pred_val / loader_size, epoch_loss_lin_val / loader_size, epoch_kl_val / loader_size

    def train(self, setting):
        train_context, _, train_loader = self._get_data(flag='train')
        _, _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.epochs):
            self.model.train()
            t = time.time()
            epoch_loss_pred = 0
            epoch_loss_lin = 0
            epoch_kl = 0

            for batch_idx, data in enumerate(train_loader):
                context_size = 10
                index_x = torch.LongTensor(random.sample(range(train_context.shape[0] - 1), context_size))
                sub_context = torch.index_select(train_context, 0, index_x)
                X_context, Y_context = sub_context[:, 0:self.args.time_shifts, :, :], sub_context[:,
                                                                                      1:self.args.time_shifts + 1, :, :]
                X_context, Y_context = X_context.to(self.device), Y_context.to(self.device)
                mu_context, sigma_context = self.model.q_cal(X_context, Y_context)
                q_context = Normal(mu_context, sigma_context)

                data = data.float()
                data = data.to(self.device)  # (batch_size, time_shifts+1, dimensions, embedding_window)
                X, Y = data[:, 0:self.args.time_shifts, :, :], data[:, 1:self.args.time_shifts + 1, :, :]

                X, Y = X.to(self.device), Y.to(self.device)
                mu_target, sigma_target = self.model.q_cal(X, Y)
                q_target = Normal(mu_target, sigma_target)
                q_target_resample = q_target.rsample()
                q_target_resample = q_target_resample.repeat(torch.cuda.device_count(), 1)

                y, x_list_mu, x_list_sigma, p_list, y_list = self.model(data, self.args.time_shifts,
                                                                        self.args.time_shifts - 1, self.args.delta_t,
                                                                        q_target_resample)

                if self.args.get_back:
                    x_list_mu, x_list_sigma, y = get_back(x_list_mu, x_list_sigma, y)
                    x_prob = Normal(x_list_mu, x_list_sigma)
                    x_sample = x_prob.rsample()
                else:
                    x_prob = Normal(x_list_mu, x_list_sigma)
                    x_sample = x_prob.rsample()

                loss_pred = 0
                if self.args.use_prob:
                    loss_pred = loss_pred - x_prob.log_prob(y).mean()
                else:
                    loss_pred = nn.MSELoss()(x_sample, y)
                loss_lin = nn.MSELoss()(p_list, y_list)
                kl = kl_divergence(q_context, q_target).mean()
                loss = loss_pred + loss_lin + kl

                epoch_loss_pred += loss_pred.item()
                epoch_loss_lin += loss_lin.item()
                epoch_kl += kl.item()

                model_optim.zero_grad()
                loss.backward(retain_graph=True)
                model_optim.step()

            epoch_loss_pred_val, epoch_loss_lin_val, epoch_kl_val = self.vali(vali_loader, q_context)

            epoch_loss_pred = epoch_loss_pred / train_steps
            epoch_loss_lin = epoch_loss_lin / train_steps
            epoch_kl = epoch_kl / train_steps

            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss_pred=", "{:.5f}".format(epoch_loss_pred), "train_loss_lin=",
                  "{:.5f}".format(epoch_loss_lin), \
                  "train_loss_KL=", "{:.5f}".format(epoch_kl), '\n',
                  "valid_loss_pred=", "{:.5f}".format(epoch_loss_pred_val), "valid_loss_lin=",
                  "{:.5f}".format(epoch_loss_lin_val), \
                  "valid_loss_KL=", "{:.5f}".format(epoch_kl_val), "time=", "{:.5f}".format(time.time() - t))

            if epoch > self.args.use_early_stop:
                early_stopping(epoch_loss_pred_val, self.model, q_context, path=path)
                if early_stopping.early_stop:
                    print("Early stopping!!!!!!!")
                    break

    def metric(self, setting, fg):
        _, origin, testloader = self._get_data(flag=fg)

        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        q_context = np.load('./checkpoints/' + setting + '/q_context.npz', allow_pickle=True)['q_context']
        q_context_resample = q_context.item().sample([testloader.batch_size])
        print('load successful')

        origin = origin[self.args.window:]
        self.model.eval()

        sigma_all = []
        mu_all = []
        for batch_idx, data in enumerate(testloader):
            data = data.float()
            data = data.to(self.device)
            y, mu_list, sigma_list = self.model.predict(data, time_shifts=self.args.predict_length,
                                                        delta_t=self.args.delta_t, q=q_context_resample)
            for m in mu_list:
                m = m.data.cpu().detach().numpy()
                mu_all.append(m)
            for s in sigma_list:
                s = s.data.cpu().detach().numpy()
                sigma_all.append(s)
            sigma_all = np.array(sigma_all)
            mu_all = np.array(mu_all)
            st = 0
            for i in range(mu_all.shape[0] - self.args.predict_length):
                i = i + st
                truth = origin[i:i + self.args.predict_length]
                mu = mu_all[i, :, :, self.args.window - 1]
                sigma = sigma_all[i, :, :, self.args.window - 1]
                crps_sum, nrmse = get_metric(truth, mu, sigma)
                break
        return crps_sum, nrmse

    def plot_interval(self, setting, fg):
        _, origin, testloader = self._get_data(flag=fg)

        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        q_context = np.load('./checkpoints/' + setting + '/q_context.npz', allow_pickle=True)['q_context']
        q_context_resample = q_context.item().sample([testloader.batch_size])
        print('loaded successful')

        origin = origin[self.args.window:]
        self.model.eval()

        sigma_all = []
        mu_all = []
        for batch_idx, data in enumerate(testloader):
            data = data.float()
            data = data.to(self.device)
            data = data
            y, mu_list, sigma_list = self.model.predict(data, time_shifts=self.args.predict_length,
                                                        delta_t=self.args.delta_t, q=q_context_resample)
            for m in mu_list:
                m = m.data.cpu().detach().numpy()
                mu_all.append(m)
            for s in sigma_list:
                s = s.data.cpu().detach().numpy()
                sigma_all.append(s)
            sigma_all = np.array(sigma_all)
            mu_all = np.array(mu_all)
            st = 0
            for i in range(mu_all.shape[0] - self.args.predict_length):
                i = i + st
                truth = origin[i:i + self.args.predict_length]
                mu = mu_all[i, :, :, self.args.window - 1]
                sigma = sigma_all[i, :, :, self.args.window - 1]
                plot_interval(truth, mu, sigma)
                break
