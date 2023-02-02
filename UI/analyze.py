import matplotlib.pyplot as plt

def analyze(ekf, x_ekf, x_true):
    x_ekf = x_ekf.detach().cpu()
    plt.plot(x_ekf[0,:,0], x_ekf[0,:,1], marker='o')
    plt.plot(x_true[0,:,0], x_true[0,:,1], marker='^')
    plt.legend(["ekf", "real"])
    plt.title("x")
    plt.show()