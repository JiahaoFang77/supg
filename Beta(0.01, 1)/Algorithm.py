import numpy as np

# 4.1 Operational Architecture

# Algorithm 1 SUPG query processing
def supg_query(D, A, O, s, gamma, delta, target_type):
    S_set = np.random.choice(range(D), size=s, replace=False)
    
    if (target_type == "RECALL"):
        tau = tau_IS_CI_R(D, A, O, s, gamma, delta)  

    if (target_type == "PRECISION"):
        tau = tau_IS_CI_P(D, A, O, s, gamma, delta)  
    
    # R1 ← {x : x ∈ S ∧ O(x) = 1}
    # R2 ← {x : x ∈ D ∧ A(x) ≥ τ 
    R1 = [x for x in S_set if O[x] == 1]
    if (tau != None):
        R2 = [x for x in range(D) if A[x] >= tau]
    else:
        R2 = []
    # return R1 ∪ R2
    return R1 + R2

# 5.2 Guarantees through Confidence Intervals
def UB(mu, sigma, s, delta):
    return mu + sigma * np.sqrt(2 / s * np.log(1/delta))

def LB(mu, sigma, s, delta):
    return mu - sigma * np.sqrt(2 / s * np.log(1/delta))

# RecallSw (τ )
def Recall_S_w(tau, proxy_scores, oracle_labels, weights):
    S_indices = np.where(proxy_scores >= tau)
    S_w = np.arange(len(proxy_scores))
    factor = np.sum(oracle_labels[S_w] * weights[S_w])
    return np.sum((oracle_labels[S_indices] == 1) * weights[S_indices]) / factor if factor != 0 else 0

# max{τ : RecallSw (τ ) ≥ γ}
def max_tau(gamma, A, O, m, tol=1e-16):
    up, down = 1, 0
    while up - down > tol:
        mid = (down + up) / 2
        if Recall_S_w(mid, A, O, m) >= gamma:
            down = mid
        else:
            up = mid        
    return down

# 5.3 Importance Sampling
# Algorithm 4 Importance threshold estimation (RT)
def tau_IS_CI_R(D, A, O, s, gamma, delta):
    # w ← {√A(x) : x ∈ D}
    weights = np.array([np.sqrt(A[x]) for x in range(D)])

    # Defensive Mixing
    weights = 0.9 * weights / np.sum(weights) + 0.1 / D

    # S ← WeightedSample(D, ~w, s)
    S_indices = np.random.choice(range(D), size=s, replace=False, p=weights)
    m = 1 / (weights * D)

    # τo ← max{τ : RecallSw (τ ) ≥ γ}
    tau_o = max_tau(gamma, A, O, m)

    z1 = np.array([int(A[i] >= tau_o) * O[i] * m[i] for i in S_indices])
    z2 = np.array([int(A[i] < tau_o) * O[i] * m[i] for i in S_indices])

    ub = UB(np.mean(z1), np.std(z1), s, delta/2)
    lb =  LB(np.mean(z2), np.std(z2), s, delta/2)
    gamma_prime = ub / (ub + lb)

    # Compute tau_0
    tau_prime = max_tau(gamma_prime, A, O, m)
    print('gamma_prime =', gamma_prime, 'tau_o =', tau_o, 'tau_prime =',tau_prime)
    return tau_prime

# Algorithm 5 Importance threshold estimation (PT)
def tau_IS_CI_P(D, A, O, s, gamma, delta):
    # Minimum step size
    m = 100

    # w ← {√A(x) : x ∈ D}
    weights = np.array([np.sqrt(A[x]) for x in range(D)])

    # Defensive Mixing
    weights = 0.9 * weights / np.sum(weights) + 0.1 / D

    # S0 ← WeightedSample(D, w, s/2) Stage 1
    S_0 = np.random.choice(range(D), size=int(s/2), replace=False, p=weights)
    m_x = 1 / (weights * D)

    # Z ← {O(x)m(x) : x ∈ S0}
    Z = np.array([O[i] * m_x[i] for i in S_0])

    # n_match ← |D| · UB(ˆμZ , ˆσZ , s/2, δ/2)
    n_match = int(D * UB(np.mean(Z), np.std(Z), s/2, delta/2))
    
    # Sort A in descending order
    A_sorted = sorted([(A[i], i) for i in range(D)], reverse=True)
    D_prime = np.array([x[1] for x in A_sorted[:int(n_match/gamma)]])  

    # A ← SortDescending({A(x) : x ∈ D})
    # D′ ← {x : A(x) ≥ A[nmatch/γ]}
    # S1 ← WeightedSample(D′, w, s/2) . Stage 2
    weights_D0 = weights[D_prime]
    weights_D0 = weights_D0 / weights_D0.sum()   
    S_1 = np.random.choice(D_prime, size=int(s/2), replace=False, p=weights_D0) 

    # AS1 = A ∩ S1
    A_S1 = np.array([A[i] for i in S_1])
    M = int(np.ceil(int(s/2)/m))

    candidates = []
    for i in range(m, int(s/2), m):
        tau = A_S1[i]
        Z = np.array([O[j] for j in S_1 if A[j] >= tau])

        # Precision Bound
        p_l = LB(np.mean(Z), np.std(Z), len(Z), (delta/(2*M)))
        if p_l > gamma:
            candidates.append(tau)
    
    if candidates:
        print("n_match =", n_match,"min_tau =", min(candidates))
        return min(candidates)
    else:
        print("No suitable candidates found.")
        return None

