from __future__ import division

import gptools
import scipy
import matplotlib
matplotlib.use('TkAgg')
import math
import numpy as np

lw = 1
ms = 4

TEXTWIDTH = 5
TEXTHEIGHT = 5

pgf_with_latex = {
    "text.usetex": True,
    "pgf.texsystem": "lualatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 11,
    "figure.titlesize": 11,
    "pgf.preamble": [
        r"\usepackage[separate-uncertainty=true,detect-all]{siunitx}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\DeclareMathOperator{\cov}{cov}",
        r"\usepackage[x11names]{xcolor}",
        r"\definecolor{MPLr}{rgb}{1,0,0}",
        r"\definecolor{MPLg}{rgb}{0,0.5,0}",
        r"\definecolor{MPLb}{rgb}{0,0,1}",
        r"\definecolor{MPLc}{rgb}{0,0.75,0.75}",
        r"\definecolor{MPLm}{rgb}{0.75,0,0.75}",
        r"\definecolor{MPLy}{rgb}{0.75,0.75,0}",
        r"\definecolor{MPLk}{rgb}{0,0,0}",
        r"\definecolor{MPLw}{rgb}{1,1,1}",
        r"\usepackage{xspace}",
        r"\newcommand{\hirexsr}{\textsc{hirex-sr}\xspace}",
        r"\newcommand{\xeus}{\textsc{xeus}\xspace}",
        r"\newcommand{\loweus}{\textsc{loweus}\xspace}",
        r"\newcommand{\xtomo}{\textsc{xtomo}\xspace}",
        r"\DeclareMathOperator{\corr}{corr}"
    ],
    "text.latex.preamble": [
        r"\usepackage[separate-uncertainty=true,detect-all]{siunitx}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\DeclareMathOperator{\cov}{cov}",
        r"\usepackage[x11names]{xcolor}",
        r"\definecolor{MPLr}{rgb}{1,0,0}",
        r"\definecolor{MPLg}{rgb}{0,0.5,0}",
        r"\definecolor{MPLb}{rgb}{0,0,1}",
        r"\definecolor{MPLc}{rgb}{0,0.75,0.75}",
        r"\definecolor{MPLm}{rgb}{0.75,0,0.75}",
        r"\definecolor{MPLy}{rgb}{0.75,0.75,0}",
        r"\definecolor{MPLk}{rgb}{0,0,0}",
        r"\definecolor{MPLw}{rgb}{1,1,1}",
        r"\usepackage{xspace}",
        r"\newcommand{\hirexsr}{\textsc{hirex-sr}\xspace}",
        r"\newcommand{\xeus}{\textsc{xeus}\xspace}",
        r"\newcommand{\loweus}{\textsc{loweus}\xspace}",
        r"\newcommand{\xtomo}{\textsc{xtomo}\xspace}",
        r"\DeclareMathOperator{\corr}{corr}"
    ],
    'legend.handlelength': 2.5,
    "image.cmap": 'plasma',
    "lines.linewidth": lw,
    "lines.markersize": ms
}

pgf_with_latex_beamer = {
    "text.usetex": True,
    "pgf.texsystem": "lualatex",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 11,
    "figure.titlesize": 11,
    "pgf.preamble": [
        r"\usepackage[separate-uncertainty=true,detect-all]{siunitx}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\DeclareMathOperator{\cov}{cov}",
        r"\usepackage[x11names]{xcolor}",
        r"\definecolor{MPLr}{rgb}{1,0,0}",
        r"\definecolor{MPLg}{rgb}{0,0.5,0}",
        r"\definecolor{MPLb}{rgb}{0,0,1}",
        r"\definecolor{MPLc}{rgb}{0,0.75,0.75}",
        r"\definecolor{MPLm}{rgb}{0.75,0,0.75}",
        r"\definecolor{MPLy}{rgb}{0.75,0.75,0}",
        r"\definecolor{MPLk}{rgb}{0,0,0}",
        r"\definecolor{MPLw}{rgb}{1,1,1}",
        r"\usepackage{xspace}",
        r"\newcommand{\hirexsr}{HiReX-SR\xspace}",
        r"\newcommand{\xeus}{XEUS\xspace}",
        r"\newcommand{\loweus}{LoWEUS\xspace}",
        r"\newcommand{\xtomo}{XTOMO\xspace}",
        r"\DeclareMathOperator{\corr}{corr}"
    ],
    "text.latex.preamble": [
        r"\usepackage[separate-uncertainty=true,detect-all]{siunitx}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\DeclareMathOperator{\cov}{cov}",
        r"\usepackage[x11names]{xcolor}",
        r"\definecolor{MPLr}{rgb}{1,0,0}",
        r"\definecolor{MPLg}{rgb}{0,0.5,0}",
        r"\definecolor{MPLb}{rgb}{0,0,1}",
        r"\definecolor{MPLc}{rgb}{0,0.75,0.75}",
        r"\definecolor{MPLm}{rgb}{0.75,0,0.75}",
        r"\definecolor{MPLy}{rgb}{0.75,0.75,0}",
        r"\definecolor{MPLk}{rgb}{0,0,0}",
        r"\definecolor{MPLw}{rgb}{1,1,1}",
        r"\usepackage{xspace}",
        r"\newcommand{\hirexsr}{HiReX-SR\xspace}",
        r"\newcommand{\xeus}{XEUS\xspace}",
        r"\newcommand{\loweus}{LoWEUS\xspace}",
        r"\newcommand{\xtomo}{XTOMO\xspace}",
        r"\DeclareMathOperator{\corr}{corr}"
    ],
    'legend.handlelength': 2.5,
    "image.cmap": 'plasma',
    "lines.linewidth": lw,
    "lines.markersize": ms,
    'savefig.transparent': True
}

iop = {
    "text.usetex": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 10,
    "figure.titlesize": 10,
    "text.latex.preamble": [
        r"\usepackage{mathptmx}"
        r"\usepackage[separate-uncertainty=true,detect-all]{siunitx}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\DeclareMathOperator{\cov}{cov}",
        r"\usepackage[x11names]{xcolor}",
        r"\definecolor{MPLr}{rgb}{1,0,0}",
        r"\definecolor{MPLg}{rgb}{0,0.5,0}",
        r"\definecolor{MPLb}{rgb}{0,0,1}",
        r"\definecolor{MPLc}{rgb}{0,0.75,0.75}",
        r"\definecolor{MPLm}{rgb}{0.75,0,0.75}",
        r"\definecolor{MPLy}{rgb}{0.75,0.75,0}",
        r"\definecolor{MPLk}{rgb}{0,0,0}",
        r"\definecolor{MPLw}{rgb}{1,1,1}",
        r"\usepackage{xspace}",
        r"\newcommand{\hirexsr}{HiReX-SR\xspace}",
        r"\newcommand{\xeus}{XEUS\xspace}",
        r"\newcommand{\loweus}{LoWEUS\xspace}",
        r"\newcommand{\xtomo}{XTOMO\xspace}",
        r"\DeclareMathOperator{\corr}{corr}"
    ],
    'legend.handlelength': 2.5,
    "image.cmap": 'plasma',
    "lines.linewidth": lw,
    "lines.markersize": ms
}

def thesis_format():
    global TEXTWIDTH
    global TEXTHEIGHT
    
    # Width of my LaTeX document, converted to inches:
    TEXTWIDTH = 30.0 * 12.0 / 72.27
    # Height of my LaTeX document, converted to inches:
    TEXTHEIGHT = 45.0 * 12.0 / 72.27
    
    matplotlib.rcParams.update(pgf_with_latex)

def slide_format():
    global TEXTWIDTH
    global TEXTHEIGHT
    
    # Width of my LaTeX document, converted to inches:
    TEXTWIDTH = 307.29 / 72.27
    # Height of my LaTeX document, converted to inches:
    TEXTHEIGHT = 259.685 / 72.27
    
    matplotlib.rcParams.update(pgf_with_latex_beamer)

def iop_format():
    global TEXTWIDTH
    global TEXTHEIGHT
    
    # Width of ONE COLUMN, in inches:
    TEXTWIDTH = 3.375
    # Height of text, in inches:
    TEXTHEIGHT = 10.0
    
    matplotlib.rcParams.update(iop)

# Functions to auto-generate tables:
def generate_post_sum(modes, samplers, burns, lead_text, header_lines=None):
    """Generate a "posterior summary" table.
    
    Parameters
    ----------
    modes : list of lists of floats
        The modes for each parameter. Each entry should be a list of modes
        corresponding to a given case.
    samplers : list of samplers
        The samplers for each case.
    burns : list of ints
        The number of samples to burn for each case.
    lead_text : list of lists of strings
        The leading column(s). Each entry should contain the text to go in the
        column(s) before the mode. Use the empty string for empty cells.
    header_lines : list of strings, optional
        If present, these lines are printed before each set of parameters.
    """
    if header_lines is None:
        header_lines = [None,] * len(modes)
    post_sum = ""
    for m, s, b, lt, hl in zip(modes, samplers, burns, lead_text, header_lines):
        post_sum += '\\midrule\n'
        if hl is not None:
            post_sum += hl + '\\\\\n'
        means, lbs, ubs = gptools.summarize_sampler(s, burn=b)
        for ltv, modev, meanv, lbv, ubv in zip(lt, m, means, lbs, ubs):
            for t in ltv:
                post_sum += t + ' & '
            post_sum += '{mode:.4g} & {mean:.4g} & [ & {lb:.4g} & {ub:.4g} & ]\\\\\n'.format(mode=modev, mean=meanv, lb=lbv, ub=ubv)
    return post_sum

def generate_latex_tabular(fmts, break_locs, *cols):
    """Generate a LaTeX tabular.
    
    Parameters
    ----------
    fmts : list of str
        Format strings for each column.
    break_locs : list of int
        Zero-based indices of rows which should have a rule placed before them.
    *cols : lists
        The values to go in each column. These will be processed through the
        format strings given in `fmts`.
    """
    out_str = ''
    for i, cvals in enumerate(zip(*cols)):
        if i in break_locs:
            out_str += '\\midrule\n'
        for f, v in zip(fmts[:-1], cvals[:-1]):
            out_str += f % v + ' & '
        out_str += fmts[-1] % cvals[-1] + '\\\\\n'
    return out_str

def generate_latex_tabular_rows(fmts, break_locs, *rows):
    """Generate a LaTeX tabular from rows.
    
    Parameters
    ----------
    fmts : list of str
        Format strings for each column.
    break_locs : list of int
        Zero-based indices of rows which should have a rule placed before them.
    *rows : lists
        The values to go in each row. These will be processed through the
        format strings given in `fmts`.
    """
    out_str = ''
    for i, r in enumerate(rows):
        if i in break_locs:
            out_str += '\\midrule\n'
        for f, v in zip(fmts[:-1], r[:-1]):
            out_str += f % v + ' & '
        out_str += fmts[-1] % r[-1] + '\\\\\n'
    return out_str

def make_pcolor_grid(grid):
    """Make grids with the correct spacing for pcolor/pcolormesh.
    """
    grid = scipy.asarray(grid, dtype=float)
    pgrid = (grid[1:] + grid[:-1]) / 2.0
    pgrid = scipy.concatenate(([grid[0] - (pgrid[0] - grid[0]),], pgrid, [grid[-1] + (grid[-1] - pgrid[-1]),]))
    return pgrid

def hist_bins(y):
    """Compute the number of bins for a histogram.
    """
    lq, uq = scipy.stats.scoreatpercentile(y, [25, 75])
    h = 2.0 * (uq - lq) / len(y)**(1.0 / 3.0)
    return int(scipy.ceil((y.max() - y.min()) / h))

class PgfScalarFormatter(matplotlib.ticker.ScalarFormatter):
    """Custom formatter class which inserts vphantom into each label, for use with oldstyle numbers.
    """
    def _set_format(self, vmin, vmax):
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = list(self.locs) + [vmin, vmax]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        # Curvilinear coordinates can yield two identical points.
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        # Both points might be zero.
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, 3 - loc_range_oom)
        # refined estimate:
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        self.format = '%1.' + str(sigfigs) + 'f'
        if self._usetex:
            self.format = r'$%s\vphantom{0123456789}$' % self.format
        elif self._useMathText:
            self.format = '$\mathdefault{%s}$' % self.format

class PgfLogFormatter(matplotlib.ticker.LogFormatter):
    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        b = self._base
        usetex = matplotlib.rcParams['text.usetex']
        
        # only label the decades
        if x == 0:
            if usetex:
                return '$0$'
            else:
                return r'$\mathdefault{0}\vphantom{10^{0123456789}}$'
        
        fx = math.log(abs(x)) / math.log(b)
        is_decade = matplotlib.ticker.is_close_to_int(fx)

        sign_string = '-' if x < 0 else ''

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        if not is_decade and self.labelOnlyBase:
            return ''
        elif not is_decade:
            if usetex:
                return (r'$%s%s^{%.2f}\vphantom{10^{0123456789}}$') % \
                                            (sign_string, base, fx)
            else:
                return (r'$\mathdefault{%s%s^{%.2f}}\vphantom{10^{0123456789}}$') % \
                                            (sign_string, base, fx)
        else:
            if usetex:
                return (r'$%s%s^{%d}\vphantom{10^{0123456789}}$') % (sign_string,
                                           base,
                                           matplotlib.ticker.nearest_long(fx))
            else:
                return (r'$\mathdefault{%s%s^{%d}}\vphantom{10^{0123456789}}$') % (sign_string,
                                                         base,
                                                         matplotlib.ticker.nearest_long(fx))

def apply_formatter(f):
    for a in f.axes:
        xfmt = a.xaxis.get_major_formatter()
        if isinstance(xfmt, matplotlib.ticker.LogFormatter):
            a.xaxis.set_major_formatter(PgfLogFormatter())
        elif isinstance(xfmt, matplotlib.ticker.ScalarFormatter):
            a.xaxis.set_major_formatter(PgfScalarFormatter())
        elif isinstance(xfmt, matplotlib.ticker.FixedFormatter):
            seq = list(xfmt.seq)
            for i in range(0, len(seq)):
                if seq[i][-1] == '$' and "vphantom" not in seq[i]:
                    seq[i] = seq[i][:-1] + r'\vphantom{0123456789}$'
            xfmt.seq = seq
        
        yfmt = a.yaxis.get_major_formatter()
        if isinstance(yfmt, matplotlib.ticker.LogFormatter):
            a.yaxis.set_major_formatter(PgfLogFormatter())
        elif isinstance(yfmt, matplotlib.ticker.ScalarFormatter):
            a.yaxis.set_major_formatter(PgfScalarFormatter())
        elif isinstance(yfmt, matplotlib.ticker.FixedFormatter):
            seq = list(yfmt.seq)
            for i in range(0, len(seq)):
                if seq[i][-1] == '$' and "vphantom" not in seq[i]:
                    seq[i] = seq[i][:-1] + r'\vphantom{0123456789}$'
            yfmt.seq = seq
            
