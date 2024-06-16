import os
import matplotlib.pyplot as plt


def relu(x):
    return max(0, x)


def prelu(x, a=0.01):
    return max(0, x) + a * min(0, x)


if __name__ == "__main__":
    x = [i for i in range(-5, 5)]

    plt.figure(figsize=(10, 5))

    # plot for ReLU
    y_relu = [relu(i) for i in x]
    plt.subplot(2, 2, 1)
    plt.plot(x, y_relu, label='ReLU')
    plt.title('ReLU Activation Function')
    plt.xlabel('x')
    plt.ylabel('ReLU(x)')
    plt.grid(True)
    plt.legend()

    # plot for PReLU
    y_prelu1 = [prelu(i, a=0.01) for i in x]
    plt.subplot(2, 2, 2)
    plt.plot(x, y_prelu1, label='PReLU', color='red')
    plt.title('PReLU Activation Function, a=0.01')
    plt.xlabel('x')
    plt.ylabel('PReLU(x, a=0.01)')
    plt.grid(True)
    plt.legend()

    # plot for PReLU
    y_prelu2 = [prelu(i, a=0.1) for i in x]
    plt.subplot(2, 2, 3)
    plt.plot(x, y_prelu2, label='PReLU', color='orange')
    plt.title('PReLU Activation Function, a=0.1')
    plt.xlabel('x')
    plt.ylabel('PReLU(x, a=0.1)')
    plt.grid(True)
    plt.legend()

    # plot for PReLU
    y_prelu3 = [prelu(i, a=0.5) for i in x]
    plt.subplot(2, 2, 4)
    plt.plot(x, y_prelu3, label='PReLU', color='yellow')
    plt.title('PReLU Activation Function, a=0.5')
    plt.xlabel('x')
    plt.ylabel('PReLU(x, a=0.5)')
    plt.grid(True)
    plt.legend()

    # show the plots
    plt.tight_layout()
    # plt.show()

    # save the plots
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\'
    plots_dir = base_dir + r"images\plots"

    plt.savefig(plots_dir + r"\activation_functions2.png")
