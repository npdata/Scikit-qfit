"""
Processing example using a Mahr file format (txt).

The code loads a file, builds the q-spectrum and displays the image.

Note that invalids are replaced with NAN.

"""

import sys
import time
import getopt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import skqfit.qspectre as qf

def load_mahr_file(filename):
    """
    loads a mahr file in txt format and returns the result as a zmap, x, y
    """

    need = ('START_X','STOP_X','START_Y','STOP_Y','INTERVAL_LIN', 'NO_TOT_POINTS')
    with open(filename,'r') as file:
        fpm={}
        line = file.readline()
        if line.strip() != "[PROFILE_HEADER]":
            raise Exception("Invalid File Format")

        # get the file parameters first
        while line.strip() != "[PROFILE_VALUES]":
            spl = line.split("=")
            if spl[0] in need:
                if len(spl) < 2:
                    raise Exception("Missing parameter: "+spl[0])
                else:
                    fpm[spl[0]] = float(spl[1])
            line = file.readline()
        if len(fpm) != len(need):
            raise Exception("Missing parameters")

        # now get the actual data
        rows = round((fpm['STOP_X']-fpm['START_X'])/fpm['INTERVAL_LIN']) + 1
        cols = round((fpm['STOP_Y']-fpm['START_Y'])/fpm['INTERVAL_LIN']) + 1
        pts =  round(float(fpm['NO_TOT_POINTS']))
        if rows * cols != pts:
            raise Exception("Unexpected number of points")

        zmap = np.zeros((rows, cols), dtype=np.float)
        x_0 = float(fpm['START_X'])
        y_0 = float(fpm['START_Y'])
        x_n = float(fpm['STOP_X'])
        y_n = float(fpm['STOP_Y'])
        spacing = float(fpm['INTERVAL_LIN'])
      
        def do_line(line):
            ll = line.split('=')
            if len(ll) == 2:
                vals = ll[1].split(' ')
                if len(vals) == 5 and int(vals[4]) == 0:
                    r = int(round((float(vals[0]) - x_0)/spacing))
                    c = int(round((float(vals[1]) - y_0)/spacing))
                    zmap[r,c] = float(vals[2])

        zmap.fill(np.nan)
        lines = file.readlines()
        for line in lines:
            do_line(line)

        x=np.linspace(x_0, x_n, rows)
        y=np.linspace(y_0, y_n, cols)

    return zmap, x, y

def disp_qspec(qmap, skip_rotnl=False, skip_interp=True, skip_low_order=False):
    """
    Display the Q-spectrum as a simple 2D map

    """
    fig, ax = plt.subplots()
    interp = 'none' if skip_interp else None      
    ax.xaxis.tick_top()
    ax.set_xlabel('Azimuthal (M)')    
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Radial (N)')
    srow = 0
    scol = 0
    if skip_rotnl:
        scol = 1
    if skip_low_order:
        srow = 1
    qm = qmap[srow:,scol:]
    peak_val = np.log10(np.max(qm))
    log_max = peak_val + 0.1
    log_min = log_max - 4.001
    offset = 10**(log_min-1)
    dmap = np.log10(qm + offset)
    im = plt.imshow(dmap, interpolation=interp)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, orientation='vertical', cax=cax2)
    plt.show()

def filter(fname, m_max=None, n_max=None):
    zmap, x, y = load_mahr_file(fname)
    qfit = qf.QSpectrum(m_max=m_max, n_max=n_max)
    qfit.data_map(x=x, y=y, zmap=zmap)
    a_nm, b_nm = qfit.q_fit(m_max, n_max)

    # Apply suitable filter in q-space
    # In this case just remove the low order azimuthal and radial
    mask = np.ones_like(a_nm)
    mask[:,0] = 0.0
    mask[0] = 0.0
    a_nm *= mask
    b_nm *= mask

    zinv = qfit.build_map(x, y, a_nm=a_nm, b_nm=b_nm, curv=0.0)

    im = plt.imshow(zinv)
    plt.colorbar(im, orientation='vertical')
    plt.show()

def residual(fname, m_max=None, n_max=None):
    zmap, x, y = load_mahr_file(fname)
    qfit = qf.QSpectrum(m_max=m_max, n_max=n_max)
    qfit.data_map(x=x, y=y, zmap=zmap)
    a_nm, b_nm = qfit.q_fit(m_max, n_max)

    zinv = qfit.build_map(x, y, a_nm=a_nm, b_nm=b_nm, extend=0.95)
    cond = zinv != 0.0
    diff = np.extract(cond, zmap - zinv)
    diff -= np.mean(diff)

    zmap.fill(np.nan)
    np.place(zmap, cond, diff)
    im = plt.imshow(zmap)
    plt.colorbar(im, orientation='vertical')
    plt.show()

def qspectrum(fname, m_max=None, n_max=None, show_rotnl=False):
    zmap, x, y = load_mahr_file(fname)
    start = time.time()
    qmap = qf.qspec(x, y, zmap, m_max=m_max, n_max=n_max)
    print('%s: spectrum done, time %.3fs' % (fname, time.time()-start))
    disp_qspec(qmap, skip_rotnl=not show_rotnl, skip_interp=True, skip_low_order=False)

if __name__ == "__main__":
    def usage():
        print('qspectrum.py --mmax=500 --nmax=500 --rotnl=False [qspec|resid|filter] filename')
    try:
        opts, args = getopt.getopt(sys.argv[1:],"",["mmax=", "nmax=", "rotnl="])
        process = args[0]
        fname = args[1]
    except (getopt.GetoptError, IndexError):
        usage()
        sys.exit(2)
    nmax = mmax = 500
    rotnl = False
    for o, a in opts:
        if o == "--mmax":
            mmax = int(a)
        elif o == "--nmax":
            nmax = int(a)
        elif o == "--rotnl":
            rotnl = (a == "True")
        else:
            usage()
            sys.exit()
    if process == "qspec":
        qspectrum(fname, m_max=mmax, n_max=nmax, show_rotnl=rotnl)
    elif process == "resid":
        residual(fname, m_max=mmax, n_max=nmax)
    elif process == "filter":
        filter(fname, m_max=mmax, n_max=nmax)
    else:
        usage()
        sys.exit()
