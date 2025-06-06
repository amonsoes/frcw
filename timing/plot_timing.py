import argparse
import csv
import matplotlib.pyplot as plt

def plot_timing(cqe_csv, ens_csv):

    root = './timing/'

    path_cqe = root + cqe_csv
    path_ens = root + ens_csv
    y_cqe_list = []
    y_ens_list = []

    with open(path_cqe, 'r') as f:
        csv_obj = csv.reader(f)
        for line in csv_obj:
            y_cqe_list.append(float(line[-1]))

    with open(path_ens, 'r') as f:
        csv_obj = csv.reader(f)
        for line in csv_obj:
            y_ens_list.append(float(line[-1]))
    
    x_axis = list(range(0, len(y_ens_list)))

    print(sum(y_cqe_list)/len(y_cqe_list))
    print(sum(y_ens_list)/len(y_ens_list))

    plt.plot(x_axis, y_ens_list, color='red', linestyle='dashed', label='Ensemble')
    plt.plot(x_axis, y_cqe_list, color='blue', linestyle='solid', label='RCW')
    plt.title("Time per iteration ")
    plt.xlabel("iteration")
    plt.ylabel("seconds")
    plt.legend()
    plt.show()

def plot_c_graph():


    c_vals = [0,0.5,1,1.5,2,2.5,3,3.5,4]
    y_cad =  [0, 0.126, 0.129, 0.131, 0.1317, 0.132, 0.1332, 0.1363,0.1349]
    y_asr = [0,0.642,0.6874,0.6774,0.674,0.6734,0.6705,0.6737,0.667]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('c')
    ax1.set_ylim(0.0,1.0)
    ax1.set_ylabel('ASR', color='red')
    ax1.plot(c_vals, y_asr, color='red', linestyle='dashed', label='ASR')

    ax2 = ax1.twinx()
    ax2.set_ylim(0.0,1.0)
    ax2.set_ylabel('CAD', color='blue')
    ax2.plot(c_vals, y_cad, color='blue', linestyle='solid', linewidth=0.7, label='CAD')

    plt.title("ASR and CAD with Increasing c Values")
    #plt.legend()
    plt.show()


def plot_lr_graph():


    lr_vals = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    y_asr =  [0,0.642,0.692,0.6975,0.6928,0.6975,0.6478,0.619]
    y_cad = [0,0.126,0.124,0.1259,0.1245,0.1259,0.276,0.385]

    fig, ax1 = plt.subplots()
    #ax1.set_xticks(lr_vals)
    ax1.set_xlabel('alpha')
    ax1.set_ylim(0.0,1.0)
    ax1.set_ylabel('ASR', color='red')
    ax1.plot(lr_vals, y_asr, color='red', linestyle='dashed', label='ASR')

    ax2 = ax1.twinx()
    ax2.set_ylabel('CAD', color='blue')
    ax2.set_ylim(0.0,1.0)
    ax2.plot(lr_vals, y_cad, color='blue', linestyle='solid', linewidth=0.7, label='CAD')

    plt.title("ASR and CAD with Increasing Alpha")
    #plt.legend()
    plt.show()


def plot_lr_graph():


    lr_vals = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    y_asr =  [0,0.642,0.692,0.6975,0.6928,0.6975,0.6478,0.619]
    y_cad = [0,0.126,0.124,0.1259,0.1245,0.1259,0.276,0.385]

    fig, ax1 = plt.subplots()
    #ax1.set_xticks(lr_vals)
    ax1.set_xlabel('alpha')
    ax1.set_ylim(0.0,1.0)
    ax1.set_ylabel('ASR', color='red')
    ax1.plot(lr_vals, y_asr, color='red', linestyle='dashed', label='ASR')

    ax2 = ax1.twinx()
    ax2.set_ylabel('CAD', color='blue')
    ax2.set_ylim(0.0,1.0)
    ax2.plot(lr_vals, y_cad, color='blue', linestyle='solid', linewidth=0.7, label='CAD')

    plt.title("ASR and CAD with Increasing Alpha")
    #plt.legend()
    plt.show()


def plot_kappa_graph():


    kappa_vals = [0,1,2,3,4,5,6,7]
    y_asr =  [0.642,0.759,0.765,0.7119,0.666,0.629,0.636,0.6109]
    y_cad = [0.126, 0.17, 0.1938, 0.195, 0.191, 0.186, 0.1909,0.1828]

    fig, ax1 = plt.subplots()
    #ax1.set_xticks(lr_vals)
    ax1.set_xlabel('kappa')
    ax1.set_ylabel('ASR', color='red')
    ax1.set_ylim(0.0,1.0)
    ax1.plot(kappa_vals, y_asr, color='red', linestyle='dashed', label='ASR')

    ax2 = ax1.twinx()
    ax2.set_ylabel('CAD', color='blue')
    ax2.set_ylim(0.0,1.0)
    ax2.plot(kappa_vals, y_cad, color='blue', linestyle='solid', linewidth=0.7, label='CAD')

    plt.title("ASR and CAD with Increasing Confidence Margins")
    #plt.legend()
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cqe_csv', type=str, default='timing_cqe_2.csv')
    parser.add_argument('--ensemble_csv', type=str, default='timing_ensemble_2.csv')
    args = parser.parse_args()


    plot_c_graph()
    plot_lr_graph()
    plot_kappa_graph()
