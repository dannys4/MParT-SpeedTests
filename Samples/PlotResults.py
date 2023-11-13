import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

res_folder = 'results'

filename_base = '../results11132023_0917/ST_CPP_eval_d5_to5_'
# quad_string = 'simpson'
quad_string = 'cc5'
backends = {
    'Serial' : 'nt1_SERIAL_{}.txt'.format(quad_string),
    'Threads (32)' : 'nt32_THREADS_{}.txt'.format(quad_string),
    'OpenMP (32)' : 'nt32_OPENMP_{}.txt'.format(quad_string),
    'Threads (8)' : 'nt8_THREADS_{}.txt'.format(quad_string),
    'OpenMP (8)' : 'nt8_OPENMP_{}.txt'.format(quad_string)
}

colors = {
    'Serial':'#377eb8', 
    'Threads (8)':'#4daf4a', 
    'Threads (32)':'#4daf4a', 
    'OpenMP (32)':'#984ea3', 
    'OpenMP (8)':'#984ea3'
}
lines = {
    'Serial':'.-', 
    'Threads (8)':'.--', 
    'Threads (32)':'.-', 
    'OpenMP (32)':'.-', 
    'OpenMP (8)':'.--'
}

num_samps = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]

for name, filename_end in backends.items():

    filename = filename_base + filename_end 
    data = np.loadtxt(filename)

    mu = np.mean(data,axis=1)/1e6
    print(name, mu[0])
    std = np.std(data,axis=1)/1e6
    plt.errorbar(num_samps, mu, yerr=std, fmt=lines[name], color=colors[name], label=name, linewidth=2)
    
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.ylabel('Computation time (s)',fontsize=12)
plt.xlabel('# samples',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

print('Saving to ', filename_base + quad_string + '_comparison.pdf')
plt.savefig(filename_base + quad_string + '_comparison.pdf',bbox_inches='tight')