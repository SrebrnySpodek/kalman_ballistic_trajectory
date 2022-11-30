#!/usr/bin/env python3

from matplotlib import pylab 
import numpy as np

# parabola movement
class Parameters():
    def __init__(self):

        # initial values
        self.x_0 = 0 
        self.y_0 = 0
        self.vx_0 = 100
        self.vy_0 = 600 # 2160 km/h
        self.a_const = 10
        self.initial_state = np.array([self.x_0, self.y_0, self.vx_0, self.vy_0, self.a_const])
        # 5 dimensional space
        self.n_dim = 5

        self.sigma_x_prediction = 100
        self.sigma_y_prediction = 1000
        self.sigma_vx_prediction = 10
        self.sigma_vy_prediction = 60
        self.sigma_a_prediction = 0.1

        self.sigma_x_measurement = 100
        self.sigma_y_measurement = 2500
        self.sigma_vx_measurement = 10
        self.sigma_vy_measurement = 60
        self.sigma_a_measurement = 0.1

        # simulation
        self.delta_t = 2
        start = 0
        end = 100
        self.time_space = np.arange(start,end, self.delta_t)
        self.nr_of_points = len(self.time_space)

def generate_measurment(parameters):

    noise_1 = np.random.normal(0,parameters.sigma_x_measurement, parameters.nr_of_points)
    noise_2 = np.random.normal(0,parameters.sigma_y_measurement, parameters.nr_of_points)
    noise_3 = np.random.normal(0,parameters.sigma_vx_measurement, parameters.nr_of_points)
    noise_4 = np.random.normal(0,parameters.sigma_vy_measurement, parameters.nr_of_points)

    xx = np.array([parameters.vx_0*t for t in parameters.time_space]) + noise_1
    yy = np.array([-parameters.a_const*t**2*0.5 + parameters.vy_0*t for t in parameters.time_space]) + noise_2
    vx = np.array([parameters.vx_0 for i in parameters.time_space]) + noise_3
    vy = np.array([parameters.vy_0 - parameters.a_const*t for t in parameters.time_space]) + noise_4

    return xx,yy,vx,vy, noise_1, noise_2, noise_3, noise_4

class Model():
    def __init__(self,paramters):
        self.D_prediction = np.array([[1,0,paramters.delta_t,0      ,0               ],
                                      [0,1,0      ,paramters.delta_t, - paramters.delta_t**2*0.5],
                                      [0,0,1      , 0     ,0               ],
                                      [0,0,0      ,1      ,- paramters.delta_t       ],
                                      [0,0,0      ,0      ,1              ]])
        self.M_measure = np.identity(paramters.n_dim)

        # uncorrelated errors -> just diagonal values
        s_d = np.array([i**2 for i in [paramters.sigma_x_prediction, paramters.sigma_y_prediction, 
            paramters.sigma_vx_prediction, paramters.sigma_vy_prediction, paramters.sigma_a_prediction]])
        s_m = np.array([i**2 for i in [paramters.sigma_x_measurement, paramters.sigma_y_measurement, 
            paramters.sigma_vx_measurement, paramters.sigma_vy_measurement, paramters.sigma_a_measurement]])

        #s_d = [0 for i in range(5)]
        self.Sigma_d = np.diagflat(s_d)
        #s_m = [0 for i in range(5)]
        self.Sigma_m = np.diagflat(s_m)
    


class Kalman():

    def __init__(self,parameters,model):

        self.P_minus = np.eye(parameters.n_dim)*0.00001 # estimated error covariance
        self.M = model.M_measure                        # measure transformation
        self.Sigma_m = model.Sigma_m                    # measure error
        self.n_dim = parameters.n_dim

    def kalman(self):
        midTerm =  np.linalg.inv(np.dot(np.dot(self.M, self.P_minus), self.M.T) + self.Sigma_m)
        K_gain = np.dot(self.P_minus, np.dot(self.M, midTerm))
        P_plus =np.dot( (np.eye(self.n_dim) - np.dot(K_gain, self.M)), self.P_minus)
        return K_gain, P_plus

def start():
    np.random.seed(0) # if, simulation each time the same 
    parameters = Parameters()
    hidden_value_y = np.array([-parameters.a_const*t**2/2 + parameters.vy_0*t for t in parameters.time_space])

    model = Model(parameters)
    kalman = Kalman(parameters, model)
    x_measurment, y_measurment, vx_measurment, vy_measurment, n1, n2, n3, n4 = generate_measurment(parameters)
    state_post = parameters.initial_state

    l_xy = []           # (x,y) predicted by model
    l_xy.append((0,0))

    for t in range(len(parameters.time_space))[1:]:
        state_predict = np.dot(model.D_prediction,state_post)
        state_measurment = np.array([x_measurment[t], y_measurment[t],
                           vx_measurment[t],vy_measurment[t],parameters.a_const]).reshape(parameters.n_dim)
        k_gain, p_plus = kalman.kalman()
        state_post = state_predict + np.dot(k_gain, (state_measurment - np.dot(model.M_measure, state_predict)))
        p_minus = np.dot(model.D_prediction, np.dot(p_plus, model.D_prediction.T)) + model.Sigma_d 
        kalman.P_minus = p_minus

        l_xy.append((state_post[0],state_post[1]))

    # plot results

    pylab.plot(parameters.time_space,[i[1] for i in l_xy],'ro',markersize = 10, label = "predicted state - kalman")
    pylab.errorbar(parameters.time_space,y_measurment,yerr = [parameters.sigma_y_measurement for i in parameters.time_space]
            ,color = 'g', fmt = 'o', markersize = 10, label = "measurment state", alpha = 0.5)
    pylab.plot(parameters.time_space,hidden_value_y,'bo', markersize = 5, label = "hidden state")

    pylab.xlabel("time", fontsize = 20)
    pylab.ylabel("height", fontsize = 20)
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    start()



