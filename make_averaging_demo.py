# Copyright 2016 Mark Chilenski
# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This script makes figure F.4 and F.5, which demonstrate the effects of various
# averaging schemes applied to Thomson scattering data.

from __future__ import division
import profiletools
import scipy

p = profiletools.neETS(1110329013, t_min=1.0, t_max=1.4, abscissa='r/a')
# Extract channel:
channel_num = 14
mask = p.channels[:, 1] == channel_num
y = p.y[mask]
err_y = p.err_y[mask]
X = p.X[mask, :]
err_X = p.err_X[mask, :]
t = p.X[mask, 0]

class Case(object):
    def __init__(self, robust, weighted, method):
        self.robust = robust
        self.weighted = weighted
        self.method = method
        
        self.mean_X, self.mean_y, self.err_X, self.err_y, T = profiletools.average_points(
            X, y, err_X, err_y, robust=self.robust, y_method=self.method, X_method=self.method, weighted=self.weighted
        )

# Evaluate the different averages, widths:
cases = []
for r in [False, True]:
    for w in [False, True]:
        if r:
            m_vals = ['sample', 'RMS', 'total']
        else:
            if w:
                m_vals = ['sample', 'RMS', 'total', 'of mean']
            else:
                m_vals = ['sample', 'RMS', 'total', 'of mean', 'of mean sample']
        for m in m_vals:
            cases += [Case(r, w, m),]

# Evaluate the empirical CDF:
# Unweighted:
eval_grid = scipy.linspace(0, 1.2, 1000)
uecdf = scipy.zeros_like(eval_grid)
data_uecdf = scipy.zeros_like(y)
for yi in y:
    uecdf[eval_grid >= yi] += 1.0
    data_uecdf[y >= yi] += 1.0
uecdf /= float(len(y))
data_uecdf /= float(len(y))
# Weighted:
wecdf = scipy.zeros_like(eval_grid)
data_wecdf = scipy.zeros_like(y)
for yi, syi in zip(y, err_y):
    wecdf[eval_grid >= yi] += syi**(-2.0)
    data_wecdf[y >= yi] += syi**(-2.0)
# Do it this way to prevent roundoff error:
normalization = wecdf[-1]
wecdf /= normalization
data_wecdf /= normalization

# Make plots:
import setupplots
setupplots.thesis_format()
import matplotlib.pyplot as plt
plt.ion()

colormap = {
    'sample': 'b',
    'RMS': 'r',
    'total': 'g',
    'of mean': 'c',
    'of mean sample': 'y'
}

# Time-history:
f_history = plt.figure(figsize=[setupplots.TEXTWIDTH, setupplots.TEXTWIDTH / 1.618])
a_history = f_history.add_subplot(1, 1, 1)
a_history.errorbar(t, y, yerr=err_y, fmt='o')
a_history.set_xlabel('$t$ [s]')
a_history.set_ylabel(r'$T_{\text{e}}$ [keV]')
a_history.set_title("edge Thomson scattering channel %d" % (channel_num,))
a_history.axhline(y=cases[0].mean_y, ls='-', color='g', label='unweighted mean')
a_history.axhline(y=cases[5].mean_y, ls='-', color='r', label='weighted mean')
a_history.axhline(y=cases[9].mean_y, ls='--', color='g', label='unweighted median')
a_history.axhline(y=cases[12].mean_y, ls='--', color='r', label='weighted median')
a_history.legend(loc='upper left', ncol=2)
a_history.set_ylim([0, 1.7])
setupplots.apply_formatter(f_history)
f_history.savefig('TS_history.pgf', bbox_inches='tight')
f_history.savefig('TS_history.pdf', bbox_inches='tight')

# Master comparison:
lw = 1.5
f = plt.figure(figsize=[setupplots.TEXTWIDTH, 6.5 / 6.0 * (3.0 / 2.0 * setupplots.TEXTWIDTH / 1.618)])
f.subplots_adjust(bottom=0.5 / 6.5 + f.subplotpars.bottom)
a_uecdf = f.add_subplot(3, 2, 1)
a_wecdf = f.add_subplot(3, 2, 2, sharex=a_uecdf, sharey=a_uecdf)
a_uhist = f.add_subplot(3, 2, 3, sharex=a_uecdf)
a_whist = f.add_subplot(3, 2, 4, sharex=a_uecdf, sharey=a_uhist)
a_uQQ = f.add_subplot(3, 2, 5, sharex=a_uecdf)
a_wQQ = f.add_subplot(3, 2, 6, sharey=a_uQQ, sharex=a_uecdf)

a_uecdf.plot(eval_grid, uecdf, lw=3)
a_wecdf.plot(eval_grid, wecdf, lw=3)
# Estimate number of bins:
lq, uq = scipy.stats.scoreatpercentile(y, [25, 75])
h = 2.0 * (uq - lq) / len(y)**(1.0 / 3.0)
n = int(scipy.ceil((y.max() - y.min()) / h))
# Plot the histograms:
a_uhist.hist(y, bins=n, normed=True)
a_whist.hist(y, bins=n, weights=(err_y)**(-2.0), normed=True)

# Plot the various cases:
for c in cases:
    a_ecdf = a_wecdf if c.weighted else a_uecdf
    a_hist = a_whist if c.weighted else a_uhist
    a_QQ = a_wQQ if c.weighted else a_uQQ
    ls = '--' if c.robust else '-'
    clr = colormap[c.method]
    if not c.weighted and not c.robust:
        lbl = c.method
    else:
        lbl = c.method + ', robust'
    if lbl[:3] == 'RMS':
        lbl = r'\textsc{rms}' + lbl[3:]
    a_ecdf.plot(eval_grid, scipy.stats.norm.cdf(eval_grid, c.mean_y, c.err_y), ls=ls, color=clr, label=lbl, lw=lw)
    if 'of mean' not in c.method:
        a_hist.plot(eval_grid, scipy.stats.norm.pdf(eval_grid, c.mean_y, c.err_y), ls=ls, color=clr, lw=lw)
        a_QQ.plot(eval_grid, (eval_grid - c.mean_y) / c.err_y, ls=ls, color=clr, label=lbl, lw=lw)
    else:
        a_hist.axvline(c.mean_y, ls=ls, color=clr, lw=lw)
        a_hist.axvspan(c.mean_y - c.err_y, c.mean_y + c.err_y, color=clr, alpha=0.25, lw=lw)
        a_QQ.axvline(c.mean_y, ls=ls, color=clr, lw=lw, label=lbl)
        a_QQ.axvspan(c.mean_y - c.err_y, c.mean_y + c.err_y, color=clr, alpha=0.25, lw=lw)

# Plot the Q-Q plots:
a_uQQ.plot(y, scipy.stats.norm.ppf(data_uecdf), 'b.', ms=12, markeredgecolor='k', markeredgewidth=0.5)
a_wQQ.plot(y, scipy.stats.norm.ppf(data_wecdf), 'b.', ms=12, markeredgecolor='k', markeredgewidth=0.5)

# a_uQQ.legend(bbox_to_anchor=(0.5, 0), bbox_transform=f.transFigure, loc='lower center', ncol=4)
a_uecdf.set_title(r'unweighted' + '\n' + r'(a) empirical \textsc{cdf}')
a_uecdf.set_ylabel(r'$\hat{F}_Y(y)$')
a_wecdf.set_title(r'weighted' + '\n' + r'(d) empirical \textsc{cdf}')
a_uhist.set_title('(b) histogram')
# a_uhist.set_xlabel('$y$ [keV]')
a_uhist.set_ylabel(r'$\hat{f}_y(y)$ [$\si{keV^{-1}}$]')
a_whist.set_title('(e) histogram')
# a_whist.set_xlabel('$y$ [keV]')
a_uQQ.set_title('(c) Q-Q plot')
a_uQQ.set_xlabel(r"$y$ [keV]")
a_uQQ.set_ylabel(r"quantiles of $\mathcal{N}(0, 1)$")
a_wQQ.set_title('(f) Q-Q plot')
a_wQQ.set_xlabel(r"$y$ [keV]")
plt.setp(a_uecdf.get_xticklabels(), visible=False)
plt.setp(a_wecdf.get_xticklabels(), visible=False)
plt.setp(a_uhist.get_xticklabels(), visible=False)
plt.setp(a_whist.get_xticklabels(), visible=False)
plt.setp(a_wecdf.get_yticklabels(), visible=False)
plt.setp(a_whist.get_yticklabels(), visible=False)
plt.setp(a_wQQ.get_yticklabels(), visible=False)
setupplots.apply_formatter(f)
f.savefig('TS_average_demo.pgf', bbox_inches='tight')
f.savefig('TS_average_demo.pdf', bbox_inches='tight')

# Put legend in seperate figure:
f_leg = plt.figure()
l = f_leg.legend(*a_uQQ.get_legend_handles_labels(), ncol=2, loc='center')
f_leg.canvas.draw()
f_leg.set_figwidth(l.get_window_extent().width / 72)
f_leg.set_figheight(l.get_window_extent().height / 72)
f_leg.savefig("legAvgDemo.pdf", bbox_inches='tight')
f_leg.savefig("legAvgDemo.pgf", bbox_inches='tight')

for c in cases:
    r = 'r' if c.robust else 'nr'
    w = 'w' if c.weighted else 'uw'
    print("%s\t%s\t%s\t%.3f\t%.3f" % (r, w, c.method, c.mean_y, c.err_y))

avg_sum = setupplots.generate_latex_tabular(
    ['%s', '%s', '%s', '%.5g', '%s', '%.5g'],
    [0, 5],
    ['Non-robust', '', '', '', '', 'Robust', '', ''],
    ['sample', r'\textsc{rms}', 'total', 'of mean', 'of mean, sample', 'sample', r'\textsc{rms}', 'total'],
    ['%.5g' % (cases[0].mean_y,), '', '', '', '', '%.5g' % (cases[9].mean_y,), '', ''],
    [c.err_y for c in cases[0:5]] + [c.err_y for c in cases[9:12]],
    ['%.5g' % (cases[5].mean_y,), '', '', '', '', '%.5g' % (cases[12].mean_y,), '', ''],
    [c.err_y for c in cases[5:9]] + [-999,] + [c.err_y for c in cases[12:]],
)
with open('average_demo.tex', 'w') as tf:
    tf.write(avg_sum)
