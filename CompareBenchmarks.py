import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

res_folder = 'results'

folders = {
    'Pre': 'results11122023_1757/',
    'Post': 'results11132023_0917/'
}

filename_base = 'ST_CPP_eval_d5_to5_'

quad_string = '_cc5'

backends = {
    ('Serial', '1') : 'nt1_SERIAL',
    ('Threads', '32') : 'nt32_THREADS',
    ('OpenMP', '32') : 'nt32_OPENMP',
    ('Threads', '8') : 'nt8_THREADS',
    ('OpenMP', '8') : 'nt8_OPENMP'
}

colors = {
    'Pre':'#000000', 
    'Post':'#377eb8',
}
shapes = {
    'Serial':'o',
    'Threads':'s',
    'OpenMP':'^',
}
lines = {
    '1':'-',
    '8':'--',
    '32':':'
}

num_samps = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]

for folder, path in folders.items():
    for item, filename_end in backends.items():
        name, threads = item
        filename = path + filename_base + filename_end + quad_string + '.txt'
        data = np.loadtxt(filename)

        mu = np.mean(data,axis=1)/1e6
        print(name, mu[0])
        std = np.std(data,axis=1)/1e6
        label = f'{folder}, {name}, {threads}'
        plt.errorbar(num_samps, mu, yerr=std, fmt=lines[threads], color=colors[folder], marker=shapes[name], label=label, linewidth=2)
    
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.ylabel('Computation time (s)',fontsize=12)
plt.xlabel('# samples',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

out_filename = 'compare' + quad_string + '_comparison.pdf'
print('Saving to ', out_filename)
plt.savefig(out_filename,bbox_inches='tight')