
def deconvolve_calcium_known(X, g=0.5, s_min=0.5):
    num_trials, num_steps, num_cells = C.shape
    S = np.zeros_like(X)
    C = np.zeros_like(X)
    for trial in range(num_trials):
        for cell in range(num_cells):
            c,s = oasis.functions.oasisAR1(X[trial, :, cell], g=g, s_min=0.5)
            S[trial, :, cell] = s.round()
            C[trial, :, cell] = c
    return S, C

def deconvolve_calcium_unknown(X, fps, tau_init, trainp = 0.5, snr_thresh=3):
    '''
    Deconvolve calcium traces to spikes
    '''
    
    Y = []
    S = []
    C = []
    B = []
    G = []
    T = []
    L = []
    M = []
    R = []
    D = []
    for x in X.T:
        x_train = x[:int(trainp * len(x))]
        x_valid = x[int(trainp * len(x)):]
        c, s, b, g, lam = oasis.functions.deconvolve(x_train, penalty=1, g=[np.exp(-1/(tau_init * fps))], optimize_g=20)
        sn = (x_train-c).std(ddof=1)
        c, s, b, g, lam = oasis.functions.deconvolve(x_train, penalty=1, g=[g], optimize_g=20, sn=sn)
        sn = (x_train-c).std(ddof=1)
        c, s = oasis.oasis_methods.oasisAR1(x_train-b, g=g, lam=lam, s_min=sn*snr_thresh)
        r_train = np.corrcoef(c, x_train)[0, 1]
        c, s = oasis.oasis_methods.oasisAR1(x_valid-b, g=g, lam=lam, s_min=sn*snr_thresh)
        r_valid = np.corrcoef(c, x_valid)[0, 1]
        
        c, s = oasis.oasis_methods.oasisAR1(x-b, g=g, lam=lam, s_min=sn*snr_thresh)
        
        if not np.any(np.isnan([r_train, r_valid])):
            Y.append(x)
            S.append(np.round(s/(sn*snr_thresh)))
            C.append(c)
            B.append(b)
            G.append(g)
            T.append(-1/(np.log(g)*fps))
            L.append(lam)
            M.append(sn*snr_thresh)
            R.append(r_valid)
            D.append(r_train - r_valid)
            
    def z_score(x):
        return (x - x.mean())/x.std(ddof=1)
    
    Y = np.array(Y).T
    S = np.array(S).T
    C = np.array(C).T
    B = np.array(B)
    G = np.array(G) 
    T = np.array(T) 
    L = np.array(L)
    M = np.array(M)
    R = np.array(R)
    D = np.array(D)
    
    conds = np.ones(Y.shape[1])
    conds *= (np.abs(z_score(G)) < 3)
    conds *= (np.abs(z_score(L)) < 3)
    conds *= (np.abs(z_score(T)) < 3)
    conds = conds.astype('bool')

    Y = Y[:, conds]
    S = S[:, conds]
    C = C[:, conds]
    
    B = B[conds]
    T = T[conds]
    M = M[conds]
    R = R[conds]
    
    return Y, S, C, B, T, M, R

if __name__ == '__main__':
    main()