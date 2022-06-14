import numpy as np 
from matplotlib import pyplot as plt
from tqdm import tqdm 

''' MAIN SERIES AND NEWTON ITERATION '''

def gen_series(C, G, N):
    a = np.zeros(N+1)
    a[0], a[1], a[2] = C, G, G**2/(4*C)
    for n in range(2, N):
        a[n+1] = (sum((k+1)*(k+2)*a[k+2]*a[n-k-1] for k in range(0, n-1)) + sum((k+1)*a[k+1]*a[n-k] for k in range(0, n)))/(C*n*(n+1)**2)
    return a 

def newton_iter(C, G, N):
    a = gen_series(C, G, N)
    phi_0 = sum(a)
    phi_1 = (2/C) + sum(n*a[n] for n in range(N+1))
    phi = np.array([phi_0, phi_1])
    d_phi_0_d_C = sum((1-n)*a[n] for n in range(N+1))/C
    d_phi_0_d_G = sum(n*a[n] for n in range(N+1))/G 
    d_phi_1_d_C = (-2/C**2) + sum(n*(1-n)*a[n] for n in range(N+1))/C
    d_phi_1_d_G = sum(a[n]*n**2 for n in range(N+1))/G 
    J = np.array([[d_phi_0_d_C, d_phi_0_d_G], 
                  [d_phi_1_d_C, d_phi_1_d_G]])
    deltaC, deltaG = np.linalg.solve(J, phi)
    return (C - deltaC, G - deltaG)

def run_newton_plain(C0, G0, N, num_iter):
    (C, G) = (C0, G0)
    for _ in range(num_iter):
        (C, G) = newton_iter(C, G, N)
    return (C, G)


''' CODE ADDED FOR CHECKING INTEGRATION '''
def get_tdz_estimator(N, T = 500):
    (C, G) = run_newton_plain(1.6, -2.1, T, 100)
    a = gen_series(C, G, N)
    def est_tdz(Pr):
        a_hat = np.zeros(N+1)
        a_hat[0] = 0
        for k in range(1, N+1):
            a_hat[k] = Pr * a[k] / (k * C)
        b = np.zeros(N+1)
        b[0] = 1
        for k in range(0, N):
            b[k+1] = (1/(k+1)) * sum((j+1)*a_hat[j+1]*b[k-j] for j in range(k+1))
        m_inv = (2/C) * sum(b[n]/(n+Pr) for n in range(N+1))
        m = 1/m_inv #this is the reciprocal
        fac = np.exp(Pr * sum(a[n]/(C*n) for n in range(1, N+1)))
        tdz = m * fac
        return tdz
    return est_tdz

for N in [100, 200, 400]:
    r_tdz_est = get_tdz_estimator(N)
    prs = np.linspace(.001, 14, num=100)
    tdzs = [r_tdz_est(pr) for pr in tqdm(prs)]
    plt.plot(prs, tdzs, label='N=' + str(N))

def compare_plots(Pr, N=100):
    #get series
    (C, G) = run_newton_plain(1.6, -2.1, N, 50)
    a = gen_series(C, G, N)
    #normalize constants in
    a_hat = np.zeros(N+1)
    a_hat[0] = 0
    for k in range(1, N+1):
        a_hat[k] = Pr * a[k] / (k * C)
    b = np.zeros(N+1)
    b[0] = 1
    for k in range(0, N):
        b[k+1] = (1/(k+1)) * sum((j+1)*a_hat[j+1]*b[k-j] for j in range(k+1))
    def int_funct_raw(u):
        return u ** (Pr - 1) * np.exp(sum(ah*u**n for (n, ah) in enumerate(a_hat)))
    def int_funct_expd(u):
        return sum(bv*(u**(Pr+n-1)) for (n, bv) in enumerate(b))
    us = np.linspace(0, 1, num=50, endpoint=False)
    ir = [int_funct_raw(u) for u in us]
    ie = [int_funct_expd(u) for u in us]
    plt.plot(us, ir, label='raw')
    plt.plot(us, ie, label='exp')

def check_exp_process(Pr, N=100):
    #get series
    (C, G) = run_newton_plain(1.6, -2.1, N, 50)
    a = gen_series(C, G, N)
    #normalize constants in
    a_hat = np.zeros(N+1)
    a_hat[0] = 0
    for k in range(1, N+1):
        a_hat[k] = Pr * a[k] / (k * C)
    #get exponential
    b = np.zeros(N+1)
    b[0] = 1
    for k in range(0, N):
        b[k+1] = (1/(k+1)) * sum((j+1)*a_hat[j+1]*b[k-j] for j in range(k+1))
    #get exp at 0 with pure Taylor
    prefac = 1
    br = np.zeros(N+1)
    a_pow = np.zeros(N+1)
    a_pow[0] = 1
    for k in tqdm(range(N+1)):
        br += prefac * a_pow
        prefac /= (k+1)
        a_next = np.zeros(N+1)
        for j in range(N+1):
            a_next[j] = sum(a_pow[l]*a_hat[j-l] for l in range(j+1))
        a_pow = a_next
    print(br[:18])
    print(b[:18])

plt.legend()
plt.show()