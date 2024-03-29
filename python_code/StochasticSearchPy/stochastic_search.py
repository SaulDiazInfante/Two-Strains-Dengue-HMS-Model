import numpy as np
import datetime
import yaml
from scipy import integrate
import matplotlib.pyplot as plt


class StochasticSearch():
    def __init__(self):
        self.number_of_samples = 30
        self.frecuency_per_week_DF = \
            np.loadtxt('./data/frecuency_per_week_DF.dat', dtype='int',
                       delimiter=',')
        self.frecuency_per_week_DHF = \
            np.loadtxt('./data/frecuency_per_week_DHF.dat', dtype='int',
                       delimiter=',')
        self.bound_error_FD = 100
        self.bound_error_FHD = 30
        self.bound_initial_error_FHD = 20
        self.bound_initial_error_FD = 10
        # Numerics initial parameters
        self.Lambda_M = 41933.0 * 7.0
        self.beta_M = 0.001372700
        self.beta_H = 0.042837955
        self.b = 3.807142 * 7.0
        self.mu_M = 0.07521661385714286 * 7.0
        self.mu_H = 0.000039 * 7.0
        self.alpha_c = 0.059245826 * 7.0
        self.alpha_h = 0.132552790 * 7.0
        self.sigma = 0.42264415014285717 * 7.0
        self.p = 0.050000
        self.theta = 0.086764968
        self.z_max = 1000
        #
        #
        self.M_s0 = 120000.000000
        self.M_10 = 20.000000
        self.M_20 = 30.000000
        #
        #
        self.I_10 = 10.000000
        self.I_20 = 20.000000
        self.S_0 = 35600.000000 - (self.I_10 + self.I_20)
        self.S_m1_0 = 4400.000000
        self.Y_m1_c0 = 0.0
        self.Y_m1_h0 = 0.0
        self.Rec_0 = 0.0
        self.z0 = 1.050000
        self.N_H = self.S_0 + self.I_10 + self.I_20 + self.S_m1_0
        self.N_sm1 = self.S_m1_0 + self.Y_m1_c0 + self.Y_m1_h0
        self.Lambda_S_m1 = 0.1 * self.mu_H * self.N_H
        self.Lambda_S = 0.9 * self.mu_H * self.N_H
        self.t0 = 25  # 25.0
        self.T = 53
        self.grid_size = np.int(self.T - self.t0) * 10000
        self.h = np.float64(self.T) / np.float64(self.grid_size)
        self.r_01 = 0.0
        self.r_02 = 0.0
        self.r_zero = 0
        #
        self.t = np.linspace(self.t0, self.T, self.grid_size)
        self.solution = np.zeros([len(self.t), 13])
        #
        self.fitting_error_DF = 0.0
        self.fitting_error_DHF = 0.0
        self.r_zero_cond = False
        self.fitting_error_DF_cond = False
        self.fitting_error_DHF_cond = False
        self.stop_condition = False

    def update_conditions_search(self):
        r_zero_cond = (self.r_zero > 1)
        fitting_error_DF_cond = (self.fitting_error_DF < self.bound_error_FD)
        fitting_error_DHF_cond = (self.fitting_error_DHF
                                  < self.bound_error_FHD)
        max_infected_class_cond = (self.z_max < 700)
        stop_condition = (fitting_error_DF_cond and fitting_error_DHF_cond) \
                         and r_zero_cond and max_infected_class_cond

        self.stop_condition = stop_condition
        self.r_zero_cond = r_zero_cond
        self.fitting_error_DF_cond = fitting_error_DF_cond
        self.fitting_error_DHF_cond = fitting_error_DHF_cond
        return stop_condition

    def fitting_plot(self):
        t = self.t
        Y_m1_h = self.solution[:, 8]
        z = self.solution[:, 9]
        #
        t_data_DF = self.frecuency_per_week_DF[3:, 0]
        t_data_DHF = self.frecuency_per_week_DHF[1:, 0]
        offset = 10000
        #
        t_z = t[0: -1: offset]
        t_z = np.round(t_z)
        t_z = t_z.astype(int)
        z_points = z[0:-1: offset]

        #
        delete_index_t_z = [0, 1, 6, 9]
        t_z = np.delete(t_z, delete_index_t_z)
        z_points = np.delete(z_points, delete_index_t_z)
        #
        t_Y_m1_h = t[0: -1: offset]
        t_Y_m1_h = np.round(t_Y_m1_h)
        t_Y_m1_h = t_Y_m1_h.astype(int)
        delte_index_t_Y = [0, 1, 3, 4, 6, 7, 8]
        t_Y_m1_h = np.delete(t_Y_m1_h, delte_index_t_Y)
        Y_m1_h_points = Y_m1_h[0: -1: offset]
        Y_m1_h_points = np.delete(Y_m1_h_points, delte_index_t_Y)

        #
        frecuency_per_week_DF = self.frecuency_per_week_DF[3:, 1]
        frecuency_per_week_DHF = self.frecuency_per_week_DHF[1:, 1]
        '''
        fitting_error_DF = \
            np.linalg.norm(frecuency_per_week_DF - z_points, ord=np.inf)
        self.fitting_error_DF = fitting_error_DF
        fitting_error_DHF = \
            np.linalg.norm(frecuency_per_week_DHF - Y_m1_h_points, ord=np.inf)
        self.fitting_error_DHF = fitting_error_DHF
        '''
        #
        f1, ax_array = plt.subplots(2, 2, sharex=True)
        ax_array[0, 0].plot(t, z, 'b-')
        ax_array[0, 0].set_title(r'Reported DF ')

        ax_array[0, 1].plot(t_data_DF, frecuency_per_week_DF,
                            ls='--',
                            color='lightblue',
                            marker='o',
                            ms=8,
                            mfc='lightblue',
                            alpha=0.7)
        ax_array[0, 1].plot(t, z,
                            ls=':',
                            color='darkblue',
                            alpha=0.3)
        ax_array[0, 1].plot(t_z, z_points,
                            ls='none',
                            color='blue',
                            marker='*',
                            ms=8,
                            mfc='blue',
                            alpha=0.5)
        ax_array[0, 1].text(27, 300,
                            'err=' + str(np.round(self.fitting_error_DF, 1)),
                            fontsize=10
                            )
        ax_array[0, 1].set_ylim(0, 400)

        ax_array[0, 1].set_title(r'DF Fitting ')
        #
        ax_array[1, 0].plot(t, Y_m1_h, 'r-')
        ax_array[1, 0].set_title(r'DHF')
        ax_array[1, 1].plot(t_data_DHF, frecuency_per_week_DHF,
                            ls='--',
                            color='orange',
                            marker='o',
                            ms=8,
                            mfc='orange',
                            alpha=0.5)
        ax_array[1, 1].plot(t, Y_m1_h,
                            ls=':',
                            color='crimson',
                            alpha=0.5
                            )
        ax_array[1, 1].plot(t_Y_m1_h, Y_m1_h_points,
                            ls='none',
                            color='crimson',
                            marker='*'
                            )
        ax_array[1, 1].text(27, 50,
                            'err=' + str(np.round(self.fitting_error_DHF, 1)),
                            fontsize=10
                            )
        ax_array[1, 1].set_ylim(0, 60)
        ax_array[1, 1].set_title(r'DHF Fitting ')

        for i in np.arange(2):
            ax_array[1, i].set(xlabel='week n')
        for j in np.arange(2):
            ax_array[j, 0].set(ylabel='Individuals')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('./plots/fitting_DF_DHF.png')

    def fitting_error(self):
        #
        #
        #
        t = self.t
        Y_m1_h = self.solution[:, 8]
        z = self.solution[:, 9]
        self.z_max = np.max(z)
        phase = 12
        #
        t_data_DF = self.frecuency_per_week_DF[3:, 0]
        t_data_DHF = self.frecuency_per_week_DHF[1:, 0]
        offset = 10000
        #
        t_z = t[0: -1: offset]
        t_z = np.round(t_z)
        t_z = t_z.astype(int)
        z_points = z[0:-1: offset]
        #
        #
        delete_index_t_z = [0, 1, 6, 9]
        t_z = np.delete(t_z, delete_index_t_z)
        z_points = np.delete(z_points, delete_index_t_z)
        #
        t_Y_m1_h = t[0: -1: offset]
        t_Y_m1_h = np.round(t_Y_m1_h)
        t_Y_m1_h = t_Y_m1_h.astype(int)
        delte_index_t_Y = [0, 1, 3, 4, 6, 7, 8]
        t_Y_m1_h = np.delete(t_Y_m1_h, delte_index_t_Y)
        Y_m1_h_points = Y_m1_h[0: -1: offset]
        Y_m1_h_points = np.delete(Y_m1_h_points, delte_index_t_Y)
        #
        #
        frecuency_per_week_DF = self.frecuency_per_week_DF[3:, 1]
        frecuency_per_week_DHF = self.frecuency_per_week_DHF[1:, 1]
        fitting_error_DF = \
            np.linalg.norm(frecuency_per_week_DF[0: phase]
                           - z_points[0: phase], ord=np.inf) \
            # / np.linalg.norm(frecuency_per_week_DF[0: phase], ord=2)
        self.fitting_error_DF = fitting_error_DF
        fitting_error_DHF = \
            np.linalg.norm(frecuency_per_week_DHF[0: phase]
                           - Y_m1_h_points[0: phase], ord=np.inf)  \
            # / np.linalg.norm(frecuency_per_week_DHF[0: phase], ord=2)
        self.fitting_error_DHF = fitting_error_DHF

    @staticmethod
    def f_rhs(x, t, Lambda_M, Lambda_S, Lambda_S_m1, beta_M, beta_H, b,
              mu_M, mu_H, alpha_c, alpha_h, sigma, p, theta):
        """
        :param Lambda_M:
        :type mu_M: object
        :type alpha_h: float64
        """
        M_s = x[0]
        M_I1 = x[1]
        M_I2 = x[2]
        S = x[3]
        I_1 = x[4]
        I_2 = x[5]
        S_m1 = x[6]
        Y_m1_c = x[7]
        Y_m1_h = x[8]
        z = x[9]
        R = x[10]
        #
        #
        N_H = S + I_1 + I_2 + S_m1 + Y_m1_c + Y_m1_h + R
        c_M = (beta_M * b / N_H)
        c_H = (beta_H * b / N_H)
        A_I1 = c_M * I_1
        A_I2 = c_M * I_2
        A_Y_m1_c = c_M * Y_m1_c
        A_Y_m1_h = c_M * Y_m1_h
        B_M1 = c_H * M_I1
        B_M2 = c_H * M_I2
        #
        # rhs of the ODE
        #
        dM_s = Lambda_M - (A_I1 + A_I2 + A_Y_m1_c + A_Y_m1_h) * M_s \
               - mu_M * M_s
        dM_I1 = A_I1 * M_s - mu_M * M_I1
        dM_I2 = (A_I2 + A_Y_m1_c + A_Y_m1_h) * M_s - mu_M * M_I2
        #
        dS = Lambda_S - (B_M1 + B_M2) * S - mu_H * S
        dI_1 = B_M1 * S - (alpha_c + mu_H) * I_1
        dI_2 = B_M2 * S - (alpha_c + mu_H) * I_2
        #
        dS_m1 = Lambda_S_m1 - sigma * B_M2 * S_m1 - mu_H * S_m1
        #
        dY_m1_c = (1.0 - theta) * sigma * B_M2 * S_m1 \
                  - (alpha_c + mu_H) * Y_m1_c

        dY_m1_h = theta * sigma * B_M2 * S_m1 - (alpha_h + mu_H) * Y_m1_h
        #
        dz = p * (dI_1 + dI_2 + dY_m1_c)
        dR = alpha_c * (I_1 + I_2 + Y_m1_c) + alpha_h * Y_m1_h - mu_H * R
        dydt = np.array([dM_s, dM_I1, dM_I2,
                         dS, dI_1, dI_2,
                         dS_m1, dY_m1_c, dY_m1_h, dz, dR])
        dydt = dydt.astype('float64')
        return dydt
#
    def ode_int_solution(self):
        T = self.T
        t0 = self.t0
        t = np.linspace(t0, T, self.grid_size)
        y_0 = np.array(
            [self.M_s0, self.M_10, self.M_20,
             self.S_0, self.I_10, self.I_20,
             self.S_m1_0, self.Y_m1_c0, self.Y_m1_h0,
             self.z0, self.Rec_0])
        Lambda_M = self.Lambda_M
        Lambda_S_m1 = self.Lambda_S_m1
        Lambda_S = self.Lambda_S
        beta_M = self.beta_M
        beta_H = self.beta_H
        b = self.b
        mu_M = self.mu_M
        mu_H = self.mu_H
        alpha_c = self.alpha_c
        alpha_h = self.alpha_h
        sigma = self.sigma
        p = self.p
        theta = self.theta
        #
        #
        #
        y = integrate.odeint(self.f_rhs, y_0, t,
                             args=(Lambda_M, Lambda_S, Lambda_S_m1,
                                   beta_M, beta_H, b, mu_M, mu_H, alpha_c,
                                   alpha_h, sigma, p, theta))
        self.solution = y
        self.t = t
        return y

    def parameters_sampling(self, flag_deterministic=False):
        #
        #
        if flag_deterministic:
            Lambda_M = 41933.0 * 7.0
            beta_M = 0.001372700
            beta_H = 0.042837955
            b = 3.807142 * 7.0
            mu_M = 0.07521661385714286 * 7.0
            Lambda_S = 1.4 * 7.0  # Modify to get N_H constant
            mu_H = 0.000039 * 7.0
            alpha_c = 0.059245826 * 7.0
            alpha_h = 0.132552790 * 7.0
            Lambda_S_m1 = 0.156 * 7
            sigma = 0.42264415014285717 * 7.0
            p = 0.250000
            theta = 0.086764968
            #
            #
            M_s0 = 120000.000000
            M_10 = 20.000000
            M_20 = 30.000000
            S_0 = 35600.000000
            I_10 = 1.000000
            I_20 = 20.000000
            S_m1_0 = 4400.000000
            Y_m1_c0 = 0.0
            Y_m1_h0 = 0.0
            Rec_0 = 0.0
            z0 = 1.050000
            #
            #
            self.N_H = S_0 + I_10 + I_20 + S_m1_0
            Y_m1_c0 = 0.0
            Y_m1_h0 = 0.0
            Rec_0 = 0.0
            z0 = p * (I_10 + I_20 + Y_m1_c0)
        else:
            Lambda_M = 7 * np.abs(6000 + 2000 * np.random.randn())
            beta_M = 0.05 * np.random.rand()
            b = np.abs(2 + np.random.randn()) * 7
            mu_M = (0.033 + 0.067 * np.random.rand()) * 7
            #
            # Human constants
            #
            beta_H = 0.05 * np.random.rand()
            # for recovering
            alpha_c = (0.0556 + .0444 * np.random.rand()) * 7
            alpha_h = (0.125 + 0.125 * np.random.rand()) * 7
            sigma = 0.5 + 4.5 * np.random.rand()
            #
            p = 0.1 + .2 * np.random.rand()
            # p = .05
            theta = 0.01 + 0.5 * np.random.rand()
            #
            # Initial condition mosquitoes
            p1 = 0.8 + .1 * np.random.rand()
            pj = np.random.rand(2)
            pj_hat = 0.8 * (1.0 - p1) / pj.sum() * pj
            #
            #
            # M_s0 = p1 * (Lambda_M / mu_M)
            M_s0 = 120000
            # M_10 = pj_hat[0] * (Lambda_M / mu_M)
            M_10 = 10
            # M_20 = pj_hat[1] * (Lambda_M / mu_M)
            M_20 = 10
            #
            # Initial condition humans
            #
            N_H = 37000 + 5000 * np.random.rand()
            N_S = .9 * N_H
            self.N_H = N_H
            mu_H = self.mu_H
            Lambda_S = mu_H * N_S
            Lambda_S_m1 = mu_H * (N_H - N_S)
            # partition

            I_10 = 1.000000
            I_20 = 1.000000
            S_0 = 35600.000000 - (I_10 + I_20)
#
            Y_m1_c0 = 0.0
            Y_m1_h0 = 0.0
            S_m1_0 = 4400.000000 - (Y_m1_c0 + Y_m1_h0)
            N_H = S_0 + I_10 + I_20 + S_m1_0
            #
            #
            #
            #
            Rec_0 = 0.0
            z0 = p * (I_10 + I_20 + Y_m1_c0)
        #
        #
        # Numerical parameters
        #
        T = self.T
        h = np.float64(self.T) / np.float64(self.grid_size)
        #
        # Object parameters update
        #
        self.Lambda_M = Lambda_M
        self.Lambda_S = Lambda_S
        self.Lambda_S_m1 = Lambda_S_m1
        self.beta_M = beta_M
        self.beta_H = beta_H
        self.b = b
        self.mu_M = mu_M
        self.alpha_c = alpha_c
        self.alpha_h = alpha_h
        self.sigma = sigma
        self.p = p
        self.theta = theta
        #
        self.M_s0 = M_s0
        self.M_10 = M_10
        self.M_20 = M_20
        #
        self.S_0 = S_0
        self.I_10 = I_10
        self.I_20 = I_20
        #
        self.S_m1_0 = S_m1_0
        #
        self.Y_m1_c0 = Y_m1_c0
        self.Y_m1_h0 = Y_m1_h0
        self.Rec_0 = Rec_0
        self.z0 = z0
        #
        self.h = h
        self.T = T
        #
        new_parameters = [Lambda_M, beta_M,
                          beta_H, b, mu_M, alpha_c, alpha_h,
                          sigma, p, theta, M_s0, M_10, M_20, S_0, I_10, I_20,
                          S_m1_0, Y_m1_c0, Y_m1_h0, Rec_0, z0, h, T]
        new_parameters = np.array(new_parameters)
        return new_parameters

    def save_parameters(self,
                        file_name_prefix='./OutputParameters/parameters'):

        # load parameters
        Lambda_M = self.Lambda_M
        Lambda_S = self.Lambda_S
        Lambda_S_m1 = self.Lambda_S_m1
        beta_M = self.beta_M
        beta_H = self.beta_H
        b = self.b
        mu_M = self.mu_M
        mu_H = self.mu_H
        alpha_c = self.alpha_c
        alpha_h = self.alpha_h
        sigma = self.sigma
        p = self.p
        theta = self.theta
        M_s0 = self.M_s0
        M_10 = self.M_10
        M_20 = self.M_20
        S_0 = self.S_0
        I_10 = self.I_10
        I_20 = self.I_20
        S_m1_0 = self.S_m1_0
        Y_m1_c0 = self.Y_m1_c0
        Y_m1_h0 = self.Y_m1_h0
        Rec_0 = self.Rec_0
        z0 = self.z0
        h = self.h
        T = self.T
        r_zero = self.r_zero
        parameters = {
            'Lambda_M': Lambda_M, 'Lambda_S': Lambda_S,
            'Lambda_S_m1': Lambda_S_m1,
            'beta_M': beta_M, 'beta_H': beta_H, 'b': b,
            'mu_M': mu_M, 'mu_H': mu_H, 'alpha_c': alpha_c,
            'alpha_h': alpha_h, 'sigma': sigma, 'p': p,
            'theta': theta, 'M_s0': M_s0, 'M_10': M_10,
            'M_20': M_20, 'S_0': S_0, 'I_10': I_10,
            'I_20': I_20, 'S_m1_0': S_m1_0, 'Y_m1_c0': Y_m1_c0,
            'Y_m1_h0': Y_m1_h0,
            'Rec_0': Rec_0, 'z0': z0, 'h': h,
            'T': T, 'r_zero': r_zero
            }
        #
        #
        #
        #
        str_time = str(datetime.datetime.now())
        file_name = file_name_prefix + str_time + '.yml'
        with open(file_name, 'w') as outfile:
            yaml.dump(parameters, outfile, default_flow_style=False)

    def compute_r_zero(self):
        # load parameters
        Lambda_M = self.Lambda_M
        beta_M = self.beta_M
        beta_H = self.beta_H
        b = self.b
        mu_M = self.mu_M
        mu_H = self.mu_H
        alpha_c = self.alpha_c
        alpha_h = self.alpha_h
        sigma = self.sigma
        theta = self.theta
        S_0 = self.S_0
        S_m1_0 = self.S_m1_0
        #
        # p = self.p
        # theta = self.theta
        #
        N_H = self.N_H
        N_sm1 = self.N_sm1
        #
        #
        #
        pi_r = (beta_H * beta_M * b ** 2 * Lambda_M) / (N_H ** 2 * mu_M ** 2)
        r_01 = pi_r * ((N_H - N_sm1) + + sigma * (1.0 - theta) * N_sm1) \
               * (alpha_c + mu_H) ** (-1)
        #
        #
        #
        r_02 = pi_r * sigma * theta * N_sm1 * (alpha_c + alpha_h) ** (-1)
        #
        #
        #
        self.r_01 = r_01
        self.r_02 = r_02
        r_zero = np.sqrt(self.r_01 + self.r_02)
        self.r_zero = r_zero
        return np.sqrt(r_01), np.sqrt(r_02), r_zero

    def solution_plot(self):

        M_s = self.solution[:, 0]
        M_1 = self.solution[:, 1]
        M_2 = self.solution[:, 2]
        S = self.solution[:, 3]
        I_1 = self.solution[:, 4]
        I_2 = self.solution[:, 5]
        S_m1 = self.solution[:, 6]
        Y_m1_c = self.solution[:, 7]
        Y_m1_h = self.solution[:, 8]
        z = self.solution[:, 9]
        recovers = self.solution[:, 10]
        #
        t = self.t
        N_H = S + I_1 + I_2 + S_m1 + Y_m1_c + Y_m1_h + recovers
        #
        f1, ax_array = plt.subplots(4, 3, sharex=True)
        #
        ax_array[0, 0].plot(t, M_s)
        ax_array[0, 0].set_title(r'$M_s$')
        #
        #
        ax_array[0, 1].plot(t, M_1)
        ax_array[0, 1].set_title(r'$M_1$')
        #
        #
        ax_array[0, 2].plot(t, M_2)
        ax_array[0, 2].set_title(r'$M_2$')
        #
        #   Infected humans first time
        ax_array[1, 0].plot(t, S)
        ax_array[1, 0].set_title(r'$I_s$')
        #
        #
        ax_array[1, 1].plot(t, I_1)
        ax_array[1, 1].set_title(r'$I_1$')
        #
        #
        ax_array[1, 2].plot(t, I_2)
        ax_array[1, 2].set_title(r'$I_2$')
        ##
        ##
        ax_array[2, 0].plot(t, S_m1)
        ax_array[2, 0].set_title(r'$S_{-1}$')
        ax_array[2, 1].plot(t, Y_m1_c)
        ax_array[2, 1].set_title(r'$Y_{-1}^{[c]}$')
        #
        #
        ax_array[2, 2].plot(t, Y_m1_h)
        ax_array[2, 2].set_title(r'$Y_{-1}^{[h]}$')
        #
        #
        ax_array[3, 0].plot(t, N_H)
        ax_array[3, 0].set_title(r'$N_H$')
        #
        #
        ax_array[3, 1].plot(t, recovers)
        ax_array[3, 1].set_title(r'$R$')
        #
        ax_array[3, 2].plot(t, z)
        ax_array[3, 2].set_title(r'$z$')
        #
        for i in np.arange(3):
            ax_array[3, i].set(xlabel='time')
        for j in np.arange(4):
            ax_array[j, 0].set(ylabel='Individuals')
        #
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('./plots/populations_grid.png')
        plt.close(f1)
        #
        #
        plt.figure(2)
        #
        plt.subplot(211)
        plt.plot(t, z,
                 ls='-',
                 color='blue',
                 alpha=0.4
                 )
        plt.xlabel(r'time(weeks)')
        plt.ylabel(r'$p * (I_1 + I_2 + Y_{-1})$')
        #
        plt.subplot(212)
        plt.plot(t, Y_m1_h,
                 ls='-',
                 color='red',
                 alpha=0.4
                 )
        plt.xlabel(r'time(weeks)')
        plt.ylabel(r'$Y_{-1h}$')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # plt.show()
        plt.savefig('./plots/DF_DHF.png')
        plt.close(2)

    def load_parameters(self, file_name):
        with open(file_name, 'r') as f:
            parameter_data = yaml.load(f)
        # Set initial conditions
        #
        self.Lambda_M = np.float64(parameter_data.get('Lambda_M'))
        self.beta_M = np.float64(parameter_data.get('beta_M'))
        self.beta_H = np.float64(parameter_data.get('beta_H'))
        self.b = np.float64(parameter_data.get('b'))
        self.mu_M = np.float64(parameter_data.get('mu_M'))
        self.alpha_c = np.float64(parameter_data.get('alpha_c'))
        self.alpha_h = np.float64(parameter_data.get('alpha_h'))
        self.sigma = np.float64(parameter_data.get('sigma'))
        self.p = np.float64(parameter_data.get('p'))
        self.theta = np.float64(parameter_data.get('theta'))
        self.M_s0 = np.float64(parameter_data.get('M_s0'))
        self.M_10 = np.float64(parameter_data.get('M_10'))
        self.M_20 = np.float64(parameter_data.get('M_20'))
        #
        #
        self.S_0 = np.float64(parameter_data.get('S_0'))
        self.I_10 = np.float64(parameter_data.get('I_10'))
        self.I_20 = np.float64(parameter_data.get('I_20'))
        self.S_m1_0 = np.float64(parameter_data.get('S_m1_0'))
        self.Y_m1c_0 = np.float64(parameter_data.get('Y_m1c_0'))
        self.Y_m1h_0 = np.float64(parameter_data.get('Y_m1h_0'))
        self.Rec_0 = np.float64(parameter_data.get('Rec_0'))
        self.z0 = np.float64(parameter_data.get('z0'))
        self.h = np.float64(parameter_data.get('h'))
        self.T = np.float64(parameter_data.get('T'))
