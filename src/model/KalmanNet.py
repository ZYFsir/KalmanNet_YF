import torch
from functorch import jacrev
from torch import vmap
from torch import autograd, nn
from src.model.kalman_gain.KalmanGainPredictor import KalmanGainPredictor
from src.model.kalman_gain.KalmanGainPredictor_GRU import KalmanGainPredictor_GRU
from src.model.kalman_gain.KalmanGain import KalmanGain

from src.model.maths_model.SingerMovementModel import SingerMovementModel
from src.model.maths_model.TDOAMeasurementModel import TDOAMeasurementModel


class KalmanNet(nn.Module):
    """
    EKF中的SystemModel包含两项方程，而初始值、滤波中间值应当由滤波器进行保存。
    """

    # 先调用InitSequence进行x初始化
    # 再调用GenerateSequence进行滤波。测量值是一次性输入的
    #
    # 定义输入（测量值）为z，维度为m
    # 状态为x，维度为n

    def __init__(self, in_mult, out_mult, m, n, device, kalman_gain_model):
        super().__init__()
        self.y_previous = None
        self.predicted_state_prior = None
        self.predicted_measurement = None
        self.filtered_state = None
        self.filtered_state_previous_1_step = None
        self.filtered_state_previous_2_step = None
        self.predicted_state_prior_previous_1_step = None

        self.state_dim = m
        self.observation_dim = n
        self.state_mean = None
        self.state_covariance = None

        self.device = device
        if kalman_gain_model == 'default':
            self.kalman_gain_predictor = KalmanGainPredictor(self.observation_dim, self.state_dim, in_mult, out_mult)
        elif kalman_gain_model == 'GRU':
            self.kalman_gain_predictor = KalmanGainPredictor_GRU(self.observation_dim,
                                                                 1e3,
                                                                 self.state_dim * self.observation_dim,
                                                                 2)
        elif kalman_gain_model == 'EKF':
            self.kalman_gain_predictor = KalmanGain()
        self.movement_model = SingerMovementModel()
        self.measurement_model = TDOAMeasurementModel()

    def initialize_beliefs(self, mean, covariance):
        self.state_mean = mean
        self.state_covariance = covariance

        self.y_previous = None
        self.predicted_state_prior = None
        self.predicted_measurement = None
        self.filtered_state = None
        self.filtered_state_previous_1_step = None
        self.filtered_state_previous_2_step = None
        self.predicted_state_prior_previous_1_step = None

    def forward(self, observation, hidden_states, station, h):
        # Step 1: Prediction
        self.predicted_state_prior, self.predicted_measurement = self.predict(self.state_mean, station, h)

        # Step 2: Kalman Gain Prediction
        kalman_gain, updated_hidden_states = self.compute_kalman_gain(observation, hidden_states)
        # Step 3: Measurement Innovation
        self.innovation(observation)

        # Step 4: State Correction
        self.correct(kalman_gain)

        self.y_previous = observation.data
        self.filtered_state_previous_2_step = self.filtered_state_previous_1_step.data
        self.filtered_state_previous_1_step = self.filtered_state.data
        self.predicted_state_prior_previous_1_step = self.predicted_state_prior.data

        return self.filtered_state, updated_hidden_states

    # Predict
    def predict(self, x_previous_t, station, h):
        x_prior = self.movement_model(x_previous_t)
        y_prior = self.measurement_model(x_prior, station, h)
        return x_prior, y_prior

    def get_real_kalman_gain(self, m1x_real):
        m1x_real = m1x_real.unsqueeze(2)
        error = m1x_real - self.m1x_prior[:, 0:2, :]
        real_KG = error.squeeze(0) @ torch.pinverse(self.dy.T)
        return real_KG

    def get_loss(self, m1x_real):
        fn = nn.MSELoss()
        loss = fn(self.KG[0, 0:2, :], self.get_real_kalman_gain(m1x_real))
        return loss

    # Compute the Kalman Gain
    def compute_kalman_gain(self, y, hidden_state):
        # Construct network input
        network_input = self.construct_network_input(y)

        # Normalize network input
        normalized_input = self.normalize_network_input(network_input)

        # Calculate Kalman gain
        kalman_gain, new_hidden_state = self.kalman_gain_predictor(
            normalized_input, hidden_state
        )

        batch_size = kalman_gain.shape[0]
        KG = torch.reshape(kalman_gain, [batch_size, self.state_dim, self.observation_dim])
        return KG, new_hidden_state

    def construct_network_input(self, y):
        if self.y_previous is None:
            self.y_previous = y
        if self.filtered_state_previous_1_step is None:
            self.filtered_state_previous_1_step = self.state_mean
        if self.predicted_state_prior_previous_1_step is None:
            self.predicted_state_prior_previous_1_step = self.state_mean
        if self.filtered_state_previous_2_step is None:
            self.filtered_state_previous_2_step = self.state_mean

        observation_difference = y - self.y_previous
        observation_innovation_difference = y - self.predicted_measurement
        filtered_state_evolution_difference = self.filtered_state_previous_1_step - self.filtered_state_previous_2_step
        filtered_state_update_difference = (self.filtered_state_previous_1_step -
                                            self.predicted_state_prior_previous_1_step)

        return (
            observation_difference,
            observation_innovation_difference,
            filtered_state_evolution_difference,
            filtered_state_update_difference
        )

    def normalize_network_input(self, network_input):
        normalized_input = [self.normalize(tensor) for tensor in network_input]
        return normalized_input

    def normalize(self, tensor):
        # return func.normalize(tensor, p=2, dim=1, eps=1e-12, out=None)
        return tensor

    # Innovation
    def innovation(self, y):
        self.dy = y - self.predicted_measurement

    # Compute Posterior
    def correct(self, KG):
        INOV = torch.bmm(KG, self.dy)
        self.filtered_state = self.predicted_state_prior + INOV

    def jacobianBatch(self, x, a):
        if (a == 'ObsAcc'):
            g = self.h
            f_out = self.observation_dim
            f_in = self.state_dim
        elif (a == 'ModAcc'):
            g = self.f
            f_out = self.state_dim
            f_in = self.state_dim
        elif (a == 'ObsInacc'):
            f_out = self.observation_dim
            f_in = self.state_dim
        elif (a == 'ModInacc'):
            g = self.fInacc
            f_out = self.state_dim
            f_in = self.state_dim
        jac = vmap(jacrev(g))(x)
        jac_reshape = jac.reshape([self.batch_size, f_out, f_in])
        return jac_reshape

    def getJacobian(self, x, a):
        # if(x.size()[1] == 1):
        #     y = torch.reshape((x.T),[x.size()[0]])
        try:
            if (x.size()[1] == 1):
                y = torch.reshape((x.T), [x.size()[0]])
        except:
            y = torch.reshape((x.T), [x.size()[0]])

        if (a == 'ObsAcc'):
            g = self.h
        elif (a == 'ModAcc'):
            g = self.f
        elif (a == 'ObsInacc'):
            g = self.hInacc
        elif (a == 'ModInacc'):
            g = self.fInacc

        Jac = autograd.functional.jacobian(g, y)
        Jac = Jac.view(-1, self.state_dim)
        return Jac

    def initialize(self, input_dim, output_dim, in_mult, out_mult):
        from src.model.kalman_gain.KalmanGainPredictor import KalmanGainPredictor
        self.kalman_gain_predictor = KalmanGainPredictor(input_dim, output_dim, in_mult, out_mult)
        self.kalman_gain_predictor.initialize_parameters()

    def init_KalmanNet(self):
        self.f = self.model.f  # 运动模型
        self.f_batch = vmap(self.f)
        self.state_dim = self.model.state_dim  # 输入维度（测量值维度）
        self.Q = self.model.Q  # 运动模型噪声

        # Has to be transformed because of EKF non-linearity
        self.h = self.model.h
        self.h_batch = vmap(self.h)
        self.observation_dim = self.model.observation_dim  # 状态维度
        self.R = self.model.R  # 测量噪声

        self.m1x_prior_list = []
        self.m1x_posterior_list = []

        # self.KG_array = torch.zeros([batch_size, self.n, self.m])
        self.m1x_0 = self.model.m1x_0
        self.filtered_state = self.model.m1x_0
        self.m2x_0 = self.model.m2x_0
        self.inverse_batch = vmap(torch.linalg.inv)
        # Full knowledge about the model or partial? (Should be made more elegant)
        if (self.mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif (self.mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'
