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
    ax1.set_ylabel('ASR', color='red')
    ax1.plot(c_vals, y_asr, color='red', linestyle='dashed', label='ASR')

    ax2 = ax1.twinx()
    ax2.set_ylabel('CAD', color='blue')
    ax2.plot(c_vals, y_cad, color='blue', linestyle='solid', linewidth=0.7, label='CAD')

    plt.title("ASR and CAD with Increasing c Values")
    #plt.legend()
    plt.show()
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cqe_csv', type=str, default='timing_cqe_2.csv')
    parser.add_argument('--ensemble_csv', type=str, default='timing_ensemble_2.csv')
    args = parser.parse_args()


    plot_c_graph()
