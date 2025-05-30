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

    plt.plot(x_axis, y_ens_list, color='red', linestyle='dashed', label='Ensemble')
    plt.plot(x_axis, y_cqe_list, color='blue', linestyle='solid', label='RCW')
    plt.title("Time per iteration ")
    plt.xlabel("iteration")
    plt.ylabel("seconds")
    plt.legend()
    plt.show()
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cqe_csv', type=str, default='timing_cqe_2.csv')
    parser.add_argument('--ensemble_csv', type=str, default='timing_ensemble_2.csv')
    args = parser.parse_args()


    plot_timing(args.cqe_csv, args.ensemble_csv)
