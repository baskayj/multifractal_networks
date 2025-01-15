# TODO: from jnp to np
# TODO: docstrings and imports

def partial_kron_P_k(P, k):
    k = k + 1
    m = P.shape[0]
    result = jnp.zeros((m**k, m**k, m, m))
    
    for i in range(m):
        for j in range(m):
            E_ij = jnp.zeros((m, m))
            E_ij = E_ij.at[i, j].set(1.0)
            
            partial_p_k = jnp.zeros((m**k, m**k))
            
            for m_idx in range(k):
                kron_left = jnp.eye(1)
                for _ in range(m_idx):
                    kron_left = jnp.kron(kron_left, P)
                
                kron_right = jnp.eye(1)
                for _ in range(k - m_idx - 1):
                    kron_right = jnp.kron(kron_right, P)
                
                partial_p_k += jnp.kron(jnp.kron(kron_left, E_ij), kron_right)
            
            result = result.at[..., i, j].set(partial_p_k)
    
    return result

def partial_kron_l_k(l, k):
    k = k + 1
    m = l.shape[0]
    
    result = jnp.zeros((m**k, m))
    
    for i in range(m):
        E_i = jnp.zeros(m)
        E_i = E_i.at[i].set(1.0)
        
        partial_l_k = jnp.zeros(m**k)
        
        for m_idx in range(k):
            kron_left = jnp.ones(1)
            for _ in range(m_idx):
                kron_left = jnp.kron(kron_left, l)
            
            kron_right = jnp.ones(1)
            for _ in range(k - m_idx - 1):
                kron_right = jnp.kron(kron_right, l)
            
            partial_l_k += jnp.kron(jnp.kron(kron_left, E_i), kron_right)
        
        result = result.at[..., i].set(partial_l_k)
    
    return result

def partial_expectation_P(partial_P_k, l_k, N):
    # Compute the partial derivatives of the expectation values
    partial_expectation_P_all = N * jnp.einsum('ijab,j->iab', partial_P_k, l_k)

    return partial_expectation_P_all

def partial_expectation_l(P_k, partial_l_k, N):
# Compute the partial derivatives of the expectation values
    partial_expectation_l_all = N * jnp.einsum('ij,jb->ib', P_k, partial_l_k)

    return partial_expectation_l_all

def derivative_rho_P(d_values, expectation_d_i_k, rho_i_k_all, l_k, partial_E_P):
    # Prepare an array to store the results for each degree in d_values
    result = []
    for d, rho_i_k in zip(d_values, rho_i_k_all):
        term1 = ((d / expectation_d_i_k) - 1)
        derivative_d = jnp.einsum('i,iab,i->iab', rho_i_k, partial_E_P, term1)
        derivative_d = jnp.einsum('iab,i->ab', derivative_d, l_k)
        result.append(derivative_d)
    
    return jnp.array(result)

def derivative_rho_l(d_values, expectation_d_i_k, rho_i_k_all, l_k, partial_E_l, partial_l_k):
    # Prepare an array to store the results for each degree in d_values
    result = []
    for d, rho_i_k in zip(d_values, rho_i_k_all):
        # First term
        term1 =  ((d / expectation_d_i_k) - 1)
        result1 = jnp.einsum('i,ib,i->ib', rho_i_k, partial_E_l, term1)
        result1 = jnp.einsum('ib,i->b', result1, l_k)
        
        # Second term
        result2 = jnp.einsum('i,ib->b', rho_i_k, partial_l_k)
        
        result.append(result1 + result2)
    
    return jnp.array(result)

def MAE_gradient_P(f, rho_k, d_rho_p):
    # Calculate the gradient of the MAE with respect to P
    sign = jnp.where( f == rho_k, 0, ((f - rho_k) / jnp.abs(f - rho_k)) )
    sign = jnp.expand_dims(sign, axis=(1, 2))
    return - jnp.sum(sign * d_rho_p, axis=0)

def MAE_gradient_l(f, rho_k, d_rho_l):
    # Calculate the gradient of the MAE with respect to l
    sign = jnp.where( f == rho_k, 0, ((f - rho_k) / jnp.abs(f - rho_k)) )
    sign = jnp.expand_dims(sign, axis=1)
    return - jnp.sum(sign * d_rho_l, axis=0)

def second_derivative_rho_P(d_values, expectation_d_i_k, rho_i_k_all, l_k, partial_E_P):
    # Prepare an array to store the results for each degree in d_values
    result = []
    for d, rho_i_k in zip(d_values, rho_i_k_all):
        term1 = (d * (d - 1) / jnp.square(expectation_d_i_k)) - (2 * d / expectation_d_i_k) + 1
        second_derivative_d = jnp.einsum('i,iab,icd,i->abcd', rho_i_k, partial_E_P, partial_E_P, term1)
        second_derivative_d = jnp.einsum('abcd,i->abcd', second_derivative_d, l_k)
        result.append(jnp.sum(second_derivative_d, axis=0))
    
    return jnp.array(result)

def second_derivative_rho_l(d_values, expectation_d_i_k, rho_i_k_all, l_k, partial_l_k, partial_E_l):
    # Prepare an array to store the results for each degree in d_values
    result = []
    for d, rho_i_k in zip(d_values, rho_i_k_all):
        term1 = (d * (d - 1) / jnp.square(expectation_d_i_k)) - (2 * d / expectation_d_i_k) + 1
        second_derivative_d_1 = jnp.einsum('i,ib,ic,i->bc', rho_i_k, partial_E_l, partial_E_l, term1)
        second_derivative_d_1 = jnp.einsum('bc,i->bc', second_derivative_d_1, l_k)

        term2 = ((d / expectation_d_i_k) - 1)
        second_derivative_d_2 = jnp.einsum('i,ib,ic,i->bc', rho_i_k, partial_E_l, partial_l_k, term2)
        
        result.append(jnp.sum(second_derivative_d_1 + second_derivative_d_2, axis=0))
    
    return jnp.array(result)

def MAE_hessian_P(d_values, rho_k, f, derivative_rho_p, second_derivative_rho_p):
    delta = jnp.where( f == rho_k, 1, 0)
    sign = jnp.where( f == rho_k, 0, ((f - rho_k) / jnp.abs(f - rho_k)) )

    hessian_p = jnp.zeros_like(second_derivative_rho_p[0])
    for d_idx, d in enumerate(d_values):
        # Cross-term
        gradient_product = jnp.einsum('ab,cd->abcd', derivative_rho_p[d_idx], derivative_rho_p[d_idx])
        hessian_p += delta[d_idx] * gradient_product

        # Second-derivative term
        hessian_p += -sign[d_idx] * second_derivative_rho_p[d_idx]

    return hessian_p

def MAE_hessian_l(d_values, rho_k, f, derivative_rho_l, second_derivative_rho_l):
    delta = jnp.where( f == rho_k, 1, 0)
    sign = jnp.where( f == rho_k, 0, ((f - rho_k) / jnp.abs(f - rho_k)) )

    hessian_l = jnp.zeros_like(second_derivative_rho_l[0])
    for d_idx, d in enumerate(d_values):
        # Cross-term
        gradient_product = jnp.outer(derivative_rho_l[d_idx], derivative_rho_l[d_idx])
        hessian_l += delta[d_idx] * gradient_product

        # Second-derivative term
        hessian_l += -sign[d_idx] * second_derivative_rho_l[d_idx]

    return hessian_l