import sys
import io 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def read_files(*filenames):
    lines = {}
    for filename in filenames:
        with open(filename,'r') as fh:
            for line in fh:
                fields = line.split('\t')
                try:
                    step = int(fields[0])
                except ValueError:
                    continue
                lines[step] = line 
    return list(map(lambda i: i[1], sorted(lines.items(), key=lambda i: i[0])))


def main():
    lines = read_files(*sys.argv[1:])
    buf = io.StringIO()
    for line in lines: 
        buf.write(line)
    buf.seek(0)
    data = np.loadtxt(buf, delimiter='\t')
    cms = cm.ScalarMappable(cmap=cm.Reds)
    cms.set_clim(10, 18)
    for n in range(10, 18):
        l = 'layer_{}'.format(n-9)
        plt.plot(data[:,0], data[:,n], color=cms.to_rgba(n), label=l)
    plt.legend()
    plt.show()
    plt.plot(data[:,0], data[:,6])
    plt.show()
    plt.plot(data[:,0], data[:,3])
    plt.show()


if __name__ == '__main__':
    main()
