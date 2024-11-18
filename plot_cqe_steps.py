import matplotlib.pyplot as plt



# we are assuming that the desired setting is the maximum chosen value
#sorted_items = sorted(items, key=lambda x: x[0])
x_list = [70,80,90]
y_list = [23.4,24.5,21.8]


plt.figure(figsize=(6, 6))
plt.bar(x_list, y_list, color='skyblue')
plt.xticks(x_list)
plt.xlabel('Compression Values')
plt.ylabel('Nr')
plt.title('Number of steps necessary to reach optimal q')

path_dir = './cqe_steps.png'
plt.savefig(path_dir)
plt.show()