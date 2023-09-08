import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import distribution
from copy import deepcopy

from astropy.cosmology import Planck18
import astropy.units as u

from .utils import read_injection_file, pt_interp

distribution.enable_validation(False)

def generate_tgr_only_data(event_posteriors):
    Nobs = len(event_posteriors)
    
    print(f'Using {Nobs} events!')
    
    # Construct the event posterior arrays
    dphis = []
    bws_tgr = []
    for event_posterior in event_posteriors:
        dphis.append(event_posterior['dphi'].values)
        
        N_points = len(event_posterior['mass_1_source'])

        bw = np.std(event_posterior['dphi'].values) * N_points**(-1./5)
        bws_tgr.append(bw)
    
    bws_tgr = np.array(bws_tgr)
    dphis = np.array(dphis)
    
    return dphis, bws_tgr, Nobs
    

def make_tgr_only_model(dphis, bws_tgr, Nobs):
    
    mu_tgr = numpyro.sample('mu_tgr', dist.Uniform(-4, 4))
    sigma_tgr = numpyro.sample('sigma_tgr', dist.Uniform(0, 10))
            
    sigma_tgr_i = jnp.sqrt(jnp.square(sigma_tgr) + bws_tgr**2)
    log_wts = dist.Normal(mu_tgr, sigma_tgr_i).log_prob(dphis.T).T
    
    log_like_max_l = jnp.max(log_wts, axis=1)
    delta_log_wts = log_wts - log_like_max_l[:, None]
    log_like = log_like_max_l + jnp.log(jnp.sum(jnp.exp(delta_log_wts), axis=1))
    numpyro.factor('log_likelihood', jnp.sum(log_like))
    


def generate_data(event_posteriors, injection_file, use_tilts=False, use_tgr=False):
    
    Nobs = len(event_posteriors)
    
    print(f'Using {Nobs} events!')
    
    # Construct the event posterior arrays
    m1s = []
    qs = []
    cost1s = []
    cost2s = []
    a1s = []
    a2s = []
    zs = []
    pdraw = []
    dphis = []
    kde_weights = []
    
    BW_matrices = []
    BW_matrices_sel = []
    
    for k, event_posterior in enumerate(event_posteriors):
        
        m1s.append(event_posterior['mass_1_source'].values)
        qs.append(event_posterior['mass_ratio'].values)
        
        a1s.append(event_posterior['a_1'].values)
        a2s.append(event_posterior['a_2'].values)
        dphis.append(event_posterior['dphi'].values)
        
        cost1s.append(event_posterior['cos_tilt_1'].values)
        cost2s.append(event_posterior['cos_tilt_2'].values)
        zs.append(event_posterior['redshift'].values)
        pdraw.append(event_posterior['prior'].values)
        
        if use_tgr:
            d = 3
            if use_tilts:
                data_array = np.array([event_posterior['a_1'], event_posterior['a_2'], event_posterior['dphi'],
                                    event_posterior['mass_1_source'], event_posterior['mass_ratio'], 
                                    event_posterior['redshift'], event_posterior['cos_tilt_1'], event_posterior['cos_tilt_2']])

            else:
                data_array = np.array([event_posterior['a_1'], event_posterior['a_2'], event_posterior['dphi'],
                                    event_posterior['mass_1_source'], event_posterior['mass_ratio'], 
                                    event_posterior['redshift']])
        
        else:
            d = 2
            if use_tilts:
                data_array = np.array([event_posterior['a_1'], event_posterior['a_2'], 
                                    event_posterior['mass_1_source'], event_posterior['mass_ratio'], 
                                    event_posterior['redshift'], event_posterior['cos_tilt_1'], event_posterior['cos_tilt_2']])

            else:
                data_array = np.array([event_posterior['a_1'], event_posterior['a_2'], 
                                    event_posterior['mass_1_source'], event_posterior['mass_ratio'], 
                                    event_posterior['redshift']])

        weights_i = 1./(np.exp(-(event_posterior['a_1'] - 0.3)**2/(2*0.3**2)) * 
            np.exp(-(event_posterior['a_2'] - 0.3)**2/(2*0.3**2)))
        
        kde_weights.append(weights_i)
        
        N_eff = np.sum(weights_i)**2/np.sum(weights_i**2)
        
        full_cov_i = np.cov(data_array, aweights=weights_i)
        prec_i = np.linalg.inv(full_cov_i)[:d,:d]
        cov_i = np.linalg.inv(prec_i)
        
        BW_matrices.append(cov_i * N_eff**(-2./(4+d)))
        BW_matrices_sel.append(cov_i[:2,:2] * N_eff**(-2./(6)))
        
    BW_matrices = np.array(BW_matrices)
    BW_matrices_sel = np.array(BW_matrices_sel)
    
    # Turn into tensors
    event_data_array = np.array([
        m1s, qs, cost1s, cost2s, 
        a1s, a2s, dphis, zs, pdraw, kde_weights
    ])
    
    injection_data = read_injection_file(injection_file, use_tilts=use_tilts)
    Ndraw = int(injection_data['total_generated'])
    
    # Construct the injection arrays
    
    if use_tilts:
        injection_data_array = np.array([
            injection_data['mass_1'], injection_data['mass_ratio'], injection_data['cos_tilt_1'], injection_data['cos_tilt_2'], 
            injection_data['a_1'], injection_data['a_2'], injection_data['redshift'], injection_data['prior']])
        
    else:
        injection_data_array = np.array([
            injection_data['mass_1'], injection_data['mass_ratio'], injection_data['cos_tilt_1'], injection_data['cos_tilt_2'], 
            injection_data['a_1z'], injection_data['a_2z'], injection_data['redshift'], injection_data['prior']])
        
        
    return event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw
    

# Setting up the redshift interpolant dV_c/dz
zmax = 2.5
zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 1024))
dVdzdt_interp = 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp)

def make_joint_model(event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, use_tgr):
    
    m1s = event_data_array[0]
    qs = event_data_array[1]
    cost1s = event_data_array[2]
    cost2s = event_data_array[3]
    a1_a2s = event_data_array[4:6]
    a1_a2_dphis = event_data_array[4:7]
    zs = event_data_array[7]
    pdraw = event_data_array[8]
    
    m1s_sel = injection_data_array[0]
    qs_sel = injection_data_array[1]
    cost1s_sel = injection_data_array[2]
    cost2s_sel = injection_data_array[3]
    a1_a2_sel = injection_data_array[4:6]
    zs_sel = injection_data_array[6]
    pdraw_sel = injection_data_array[7]

    ###### BUILDING OUT THE MODEL

    alpha = numpyro.sample('alpha', dist.Uniform(-4, 12))
    beta = numpyro.sample('beta', dist.Uniform(-4, 12))
    mmin = 5 #numpyro.sample('mmin', dist.Uniform(3, 8))
    
    frac_bump = numpyro.sample('frac_bump', dist.Uniform(0,1))
    mu_bump = numpyro.sample('mu_bump', dist.Uniform(20,50))
    sigma_bump = numpyro.sample('sigma_bump', dist.Uniform(1,20))
    
    # Spin magnitude distribution parameters
    mu_spin = numpyro.sample('mu_spin', dist.Uniform(0, 0.7))
    sigma_spin = numpyro.sample('sigma_spin', dist.Uniform(0.05, 10))

    # Redshift parameters
    lamb = numpyro.sample('lamb', dist.Uniform(-30, 30))
    
    # Defining the models
    def log_m1_powerlaw_density(primary_masses):
        
        log_powerlaw_comp = -alpha*jnp.log(primary_masses) + jnp.log(jnp.greater_equal(primary_masses, mmin))
        log_norm = jax.lax.select(
            jnp.isclose(alpha, 1), 
            jnp.log(1 / jnp.log(100/mmin)), 
            jnp.log((1 - alpha)/(100 ** (1 - alpha) - mmin ** (1 - alpha))))
        
        return jnp.log((1-frac_bump)*jnp.exp(log_powerlaw_comp + log_norm) + frac_bump * jnp.exp(dist.Normal(mu_bump, sigma_bump).log_prob(primary_masses.T).T))
    
    def log_q_powerlaw_density(mass_ratios, primary_masses):
        
        low = mmin/primary_masses
        
        log_norm = jax.lax.select(
            jnp.isclose(beta, -1), 
            jnp.log(-1 / jnp.log(low)), 
            jnp.log((1 + beta) / (1 - low ** (1 + beta))))
        
        log_norm = jax.lax.select(jnp.isnan(log_norm), -np.inf*np.ones(np.shape(log_norm)), log_norm)
        
        return beta*jnp.log(mass_ratios) + log_norm + jnp.log(jnp.greater_equal(mass_ratios, low))
    
    def log_redshift_powerlaw(redshifts):
        return lamb * jnp.log1p(redshifts) + jnp.log(jnp.interp(redshifts, zinterp, dVdzdt_interp))
    
    # Evaluate the per event probabilities
    log_wts = \
        log_m1_powerlaw_density(m1s) + log_q_powerlaw_density(qs, m1s) \
        + log_redshift_powerlaw(zs) - jnp.log(pdraw)
        
    # Evaluate the selection term
    log_sel_wts = \
        log_m1_powerlaw_density(m1s_sel) + log_q_powerlaw_density(qs_sel, m1s_sel) \
        + log_redshift_powerlaw(zs_sel) - jnp.log(pdraw_sel)    
    
    
    # Adding the tilt
    if use_tilts:
        f_iso = numpyro.sample('f_iso', dist.Uniform(0, 1))
        sigma_tilt = numpyro.sample('sigma_tilt', dist.Uniform(0.05, 10))
        
        def log_tilt_density(cost1, cost2):
            return jnp.log(f_iso * 1/4 + (1-f_iso) * jnp.exp(-((cost1 - 1)**2 + (cost2 - 1)**2)/(2*jnp.square(sigma_tilt))/(2*np.pi*jnp.square(sigma_tilt))))

        log_wts += log_tilt_density(cost1s, cost2s)
        log_sel_wts += log_tilt_density(cost1s_sel, cost2s_sel)
    
    # # Handling the KDE with and without the TGR parameters
    if use_tgr:
        mu_tgr = numpyro.sample('mu_tgr', dist.Uniform(-4, 4))
        sigma_tgr = numpyro.sample('sigma_tgr', dist.Uniform(0, 10))
        
        sigma_evts = BW_matrices + jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin), jnp.square(sigma_tgr)])) # (Nobs, 3, 3)
        sigma_sel = jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)]))
        mu_evts = jnp.array([mu_spin, mu_spin, mu_tgr])
        
        logp_normal_sel = dist.MultivariateNormal(jnp.array([mu_spin, mu_spin]), sigma_sel).log_prob(a1_a2_sel.T)
        log_sel_wts += logp_normal_sel
    
        logp_normal = dist.MultivariateNormal(mu_evts, sigma_evts).log_prob(jnp.array([a1_a2_dphis.T,])).T[:,:,0]
        log_wts += logp_normal
    
    else:
        sigma_evts = BW_matrices + jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])) # (Nobs, 2, 2)
        sigma_sel = jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)]))
        mu_evts = jnp.array([mu_spin, mu_spin])
        
        logp_normal_sel = dist.MultivariateNormal(jnp.array([mu_spin, mu_spin]), sigma_sel).log_prob(a1_a2_sel.T)
        log_sel_wts += logp_normal_sel
    
        logp_normal = dist.MultivariateNormal(mu_evts, sigma_evts).log_prob(jnp.array([a1_a2s.T,])).T[:,:,0]
        log_wts += logp_normal
        
        
    # Adding the per event likelihood term
    log_like_max_l = jnp.max(log_wts, axis=1)
    delta_log_wts = log_wts - log_like_max_l[:, None]
    log_like = log_like_max_l + jnp.log(jnp.sum(jnp.exp(delta_log_wts), axis=1))
    numpyro.factor('log_likelihood', jnp.sum(log_like))
    
    # Selection effect term
    log_sel_max = jnp.max(log_sel_wts)
    delta_log_sel_wts = log_sel_wts - log_sel_max
    log_sel = log_sel_max + jnp.log(jnp.sum(jnp.exp(delta_log_sel_wts))) - jnp.log(Ndraw)
    numpyro.factor('selection', -Nobs * log_sel)
    
    # N eff cuts
    def log_smooth_neff_boundary(values, criteria):
        scaled_x = (values-criteria)/(0.05*criteria)
        return jax.lax.select(jnp.greater_equal(scaled_x, 0.0), 0.0, -jnp.power(scaled_x,10)) #jnp.log(jnp.greater(values, criteria)) #
    
    neff = jnp.exp(2*jnp.log(jnp.sum(jnp.exp(log_wts), axis=1)) - jnp.log(jnp.sum(jnp.exp(2*log_wts), axis=1)))
    min_neff = jnp.min(neff)
    numpyro.deterministic('neff', neff)
    numpyro.factor('neff_criteria', log_smooth_neff_boundary(min_neff, Nobs))
    
    log_mu2 = jnp.log(jnp.sum(jnp.exp(2*log_sel_wts))) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_sel - jnp.log(Ndraw) - log_mu2))
    neff_sel = jnp.exp(2*log_sel - log_s2)
    numpyro.deterministic('neff_sel', neff_sel)
    numpyro.factor('neff_sel_criteria', log_smooth_neff_boundary(neff_sel, 4*Nobs))


def make_gr_astro_model(event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts):
    
    m1s = event_data_array[0]
    qs = event_data_array[1]
    cost1s = event_data_array[2]
    cost2s = event_data_array[3]
    a1_a2s = event_data_array[4:6]
    a1_a2_dphis = event_data_array[4:7]
    zs = event_data_array[7]
    pdraw = event_data_array[8]
    
    m1s_sel = injection_data_array[0]
    qs_sel = injection_data_array[1]
    cost1s_sel = injection_data_array[2]
    cost2s_sel = injection_data_array[3]
    a1_a2_sel = injection_data_array[4:6]
    zs_sel = injection_data_array[6]
    pdraw_sel = injection_data_array[7]

    ###### BUILDING OUT THE MODEL

    alpha = numpyro.sample('alpha', dist.Uniform(-4, 12))
    beta = numpyro.sample('beta', dist.Uniform(-4, 12))
    mmin = 5 #numpyro.sample('mmin', dist.Uniform(3, 8))
    
    frac_bump = numpyro.sample('frac_bump', dist.Uniform(0,1))
    mu_bump = numpyro.sample('mu_bump', dist.Uniform(20,50))
    sigma_bump = numpyro.sample('sigma_bump', dist.Uniform(1,20))
    
    # Spin magnitude distribution parameters
    mu_spin = numpyro.sample('mu_spin', dist.Uniform(0, 0.7))
    sigma_spin = numpyro.sample('sigma_spin', dist.Uniform(0.05, 10))

    # Redshift parameters
    lamb = numpyro.sample('lamb', dist.Uniform(-30, 30))
    
    # Defining the models
    def log_m1_powerlaw_density(primary_masses):
        
        log_powerlaw_comp = -alpha*jnp.log(primary_masses) + jnp.log(jnp.greater_equal(primary_masses, mmin))
        log_norm = jax.lax.select(
            jnp.isclose(alpha, 1), 
            jnp.log(1 / jnp.log(100/mmin)), 
            jnp.log((1 - alpha)/(100 ** (1 - alpha) - mmin ** (1 - alpha))))
        
        return jnp.log((1-frac_bump)*jnp.exp(log_powerlaw_comp + log_norm) + frac_bump * jnp.exp(dist.Normal(mu_bump, sigma_bump).log_prob(primary_masses.T).T))
    
    
    def log_q_powerlaw_density(mass_ratios, primary_masses):
        
        low = mmin/primary_masses
        
        log_norm = jax.lax.select(
            jnp.isclose(beta, -1), 
            jnp.log(-1 / jnp.log(low)), 
            jnp.log((1 + beta) / (1 - low ** (1 + beta))))
        
        log_norm = jax.lax.select(jnp.isnan(log_norm), -np.inf*np.ones(np.shape(log_norm)), log_norm)
        
        return beta*jnp.log(mass_ratios) + log_norm + jnp.log(jnp.greater_equal(mass_ratios, low))
    
    def log_redshift_powerlaw(redshifts):
        return lamb * jnp.log1p(redshifts) + jnp.log(jnp.interp(redshifts, zinterp, dVdzdt_interp))
    
    # Evaluate the per event probabilities
    log_wts = \
        log_m1_powerlaw_density(m1s) + log_q_powerlaw_density(qs, m1s) \
        + log_redshift_powerlaw(zs) - jnp.log(pdraw)
        
    # Evaluate the selection term
    log_sel_wts = \
        log_m1_powerlaw_density(m1s_sel) + log_q_powerlaw_density(qs_sel, m1s_sel) \
        + log_redshift_powerlaw(zs_sel) - jnp.log(pdraw_sel)    
    
    
    # Adding the tilt
    if use_tilts:
        f_iso = numpyro.sample('f_iso', dist.Uniform(0, 1))
        sigma_tilt = numpyro.sample('sigma_tilt', dist.Uniform(0.05, 10))
        
        def log_tilt_density(cost1, cost2):
            return jnp.log(f_iso * 1/4 + (1-f_iso) * jnp.exp(-((cost1 - 1)**2 + (cost2 - 1)**2)/(2*jnp.square(sigma_tilt))/(2*np.pi*jnp.square(sigma_tilt))))

        log_wts += log_tilt_density(cost1s, cost2s)
        log_sel_wts += log_tilt_density(cost1s_sel, cost2s_sel)
    
    # # Handling the KDE with and without the TGR parameters
    
    mu_tgr = 0
    sigma_tgr = 0
    
    sigma_evts = BW_matrices + jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin), jnp.square(sigma_tgr)])) # (Nobs, 3, 3)
    sigma_sel = jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)]))
    mu_evts = jnp.array([mu_spin, mu_spin, mu_tgr])
    
    logp_normal_sel = dist.MultivariateNormal(jnp.array([mu_spin, mu_spin]), sigma_sel).log_prob(a1_a2_sel.T)
    log_sel_wts += logp_normal_sel

    logp_normal = dist.MultivariateNormal(mu_evts, sigma_evts).log_prob(jnp.array([a1_a2_dphis.T,])).T[:,:,0]
    log_wts += logp_normal
        
    # Adding the per event likelihood term
    log_like_max_l = jnp.max(log_wts, axis=1)
    delta_log_wts = log_wts - log_like_max_l[:, None]
    log_like = log_like_max_l + jnp.log(jnp.sum(jnp.exp(delta_log_wts), axis=1))
    numpyro.factor('log_likelihood', jnp.sum(log_like))
    
    # Selection effect term
    log_sel_max = jnp.max(log_sel_wts)
    delta_log_sel_wts = log_sel_wts - log_sel_max
    log_sel = log_sel_max + jnp.log(jnp.sum(jnp.exp(delta_log_sel_wts))) - jnp.log(Ndraw)
    numpyro.factor('selection', -Nobs * log_sel)
    
    # N eff cuts
    def log_smooth_neff_boundary(values, criteria):
        scaled_x = (values-criteria)/(0.05*criteria)
        return jax.lax.select(jnp.greater_equal(scaled_x, 0.0), 0.0, -jnp.power(scaled_x,10)) #jnp.log(jnp.greater(values, criteria)) #
    
    neff = jnp.exp(2*jnp.log(jnp.sum(jnp.exp(log_wts), axis=1)) - jnp.log(jnp.sum(jnp.exp(2*log_wts), axis=1)))
    min_neff = jnp.min(neff)
    numpyro.deterministic('neff', neff)
    numpyro.factor('neff_criteria', log_smooth_neff_boundary(min_neff, Nobs))
    
    log_mu2 = jnp.log(jnp.sum(jnp.exp(2*log_sel_wts))) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_sel - jnp.log(Ndraw) - log_mu2))
    neff_sel = jnp.exp(2*log_sel - log_s2)
    numpyro.deterministic('neff_sel', neff_sel)
    numpyro.factor('neff_sel_criteria', log_smooth_neff_boundary(neff_sel, 4*Nobs))

def make_tgr_only_graviton_model(dphis, bws_tgr, Nobs, use_sigma):
    
    mu_tgr = numpyro.sample('mu_tgr', dist.Uniform(-30, -20))
    
    if use_sigma:
        sigma_tgr = numpyro.sample('sigma_tgr', dist.Uniform(0, 10))
                
        sigma_tgr_i = jnp.sqrt(jnp.square(sigma_tgr) + bws_tgr**2)
        dist1 = dist.Normal(mu_tgr, sigma_tgr_i).log_prob(dphis.T).T
        dist2 = dist.Normal(mu_tgr, sigma_tgr_i).log_prob(-dphis.T-60).T
    else:
        dist1 = dist.Normal(mu_tgr, jnp.array(bws_tgr)).log_prob(dphis.T).T
        dist2 = dist.Normal(mu_tgr, jnp.array(bws_tgr)).log_prob(-dphis.T-60).T
    
    log_wts = jax.scipy.special.logsumexp(jnp.array([dist1, dist2]), axis=0)
    
    log_like_max_l = jnp.max(log_wts, axis=1)
    delta_log_wts = log_wts - log_like_max_l[:, None]
    log_like = log_like_max_l + jnp.log(jnp.sum(jnp.exp(delta_log_wts), axis=1))
    numpyro.factor('log_likelihood', jnp.sum(log_like))
    
def make_joint_graviton_model(event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, use_tgr, use_sigma):
    
    m1s = event_data_array[0]
    qs = event_data_array[1]
    cost1s = event_data_array[2]
    cost2s = event_data_array[3]
    a1_a2s = event_data_array[4:6]
    a1_a2_dphis = event_data_array[4:7]
    a1_a2_neg_dphis = deepcopy(event_data_array[4:7])
    a1_a2_neg_dphis[2] *= -1
    a1_a2_neg_dphis[2] -= 60
    zs = event_data_array[7]
    pdraw = event_data_array[8]
    
    m1s_sel = injection_data_array[0]
    qs_sel = injection_data_array[1]
    cost1s_sel = injection_data_array[2]
    cost2s_sel = injection_data_array[3]
    a1_a2_sel = injection_data_array[4:6]
    zs_sel = injection_data_array[6]
    pdraw_sel = injection_data_array[7]
    
    # BW matrix negative phi
    
    BW_matrices_neg = deepcopy(BW_matrices)
    if use_tgr:
        for BW_matrix in BW_matrices_neg:
            BW_matrix[0,2] *= -1; BW_matrix[1,2] *= -1
            BW_matrix[2,0] *= -1; BW_matrix[2,1] *= -1

    ###### BUILDING OUT THE MODEL

    alpha = numpyro.sample('alpha', dist.Uniform(-4, 12))
    beta = numpyro.sample('beta', dist.Uniform(-4, 12))
    mmin = 5 #numpyro.sample('mmin', dist.Uniform(3, 8))
    
    frac_bump = numpyro.sample('frac_bump', dist.Uniform(0,1))
    mu_bump = numpyro.sample('mu_bump', dist.Uniform(20,50))
    sigma_bump = numpyro.sample('sigma_bump', dist.Uniform(1,20))
    
    # Spin magnitude distribution parameters
    mu_spin = numpyro.sample('mu_spin', dist.Uniform(0, 0.7))
    sigma_spin = numpyro.sample('sigma_spin', dist.Uniform(0.05, 10))

    # Redshift parameters
    lamb = numpyro.sample('lamb', dist.Uniform(-30, 30))
    
    # Defining the models
    def log_m1_powerlaw_density(primary_masses):
        
        log_powerlaw_comp = -alpha*jnp.log(primary_masses) + jnp.log(jnp.greater_equal(primary_masses, mmin))
        log_norm = jax.lax.select(
            jnp.isclose(alpha, 1), 
            jnp.log(1 / jnp.log(100/mmin)), 
            jnp.log((1 - alpha)/(100 ** (1 - alpha) - mmin ** (1 - alpha))))
        
        return jnp.log((1-frac_bump)*jnp.exp(log_powerlaw_comp + log_norm) + frac_bump * jnp.exp(dist.Normal(mu_bump, sigma_bump).log_prob(primary_masses.T).T))
    
    
    def log_q_powerlaw_density(mass_ratios, primary_masses):
        
        low = mmin/primary_masses
        
        log_norm = jax.lax.select(
            jnp.isclose(beta, -1), 
            jnp.log(-1 / jnp.log(low)), 
            jnp.log((1 + beta) / (1 - low ** (1 + beta))))
        
        log_norm = jax.lax.select(jnp.isnan(log_norm), -np.inf*np.ones(np.shape(log_norm)), log_norm)
        
        return beta*jnp.log(mass_ratios) + log_norm + jnp.log(jnp.greater_equal(mass_ratios, low))
    
    def log_redshift_powerlaw(redshifts):
        return lamb * jnp.log1p(redshifts) + jnp.log(jnp.interp(redshifts, zinterp, dVdzdt_interp))
    
    # Evaluate the per event probabilities
    log_wts = \
        log_m1_powerlaw_density(m1s) + log_q_powerlaw_density(qs, m1s) \
        + log_redshift_powerlaw(zs) - jnp.log(pdraw)
        
    # Evaluate the selection term
    log_sel_wts = \
        log_m1_powerlaw_density(m1s_sel) + log_q_powerlaw_density(qs_sel, m1s_sel) \
        + log_redshift_powerlaw(zs_sel) - jnp.log(pdraw_sel)    
    
    
    # Adding the tilt
    if use_tilts:
        f_iso = numpyro.sample('f_iso', dist.Uniform(0, 1))
        sigma_tilt = numpyro.sample('sigma_tilt', dist.Uniform(0.05, 10))
        
        def log_tilt_density(cost1, cost2):
            return jnp.log(f_iso * 1/4 + (1-f_iso) * jnp.exp(-((cost1 - 1)**2 + (cost2 - 1)**2)/(2*jnp.square(sigma_tilt))/(2*np.pi*jnp.square(sigma_tilt))))

        log_wts += log_tilt_density(cost1s, cost2s)
        log_sel_wts += log_tilt_density(cost1s_sel, cost2s_sel)
    
    # # Handling the KDE with and without the TGR parameters
    if use_tgr:
        mu_tgr = numpyro.sample('mu_tgr', dist.Uniform(-30, -20))
        
        sigma_sel = jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)]))
        mu_evts = jnp.array([mu_spin, mu_spin, mu_tgr])
        
        logp_normal_sel = dist.MultivariateNormal(jnp.array([mu_spin, mu_spin]), sigma_sel).log_prob(a1_a2_sel.T)
        log_sel_wts += logp_normal_sel

        if use_sigma:
            sigma_tgr = numpyro.sample('sigma_tgr', dist.Uniform(0, 10))
        else:
            sigma_tgr = numpyro.sample('sigma_tgr', dist.Uniform(0,0.1))
        
        sigma_evts = BW_matrices + jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin), jnp.square(sigma_tgr)]))
        sigma_evts_neg = BW_matrices_neg + jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin), jnp.square(sigma_tgr)])) # (Nobs, 3, 3)
        logp_normal = jax.scipy.special.logsumexp(jnp.array([
            dist.MultivariateNormal(mu_evts, sigma_evts).log_prob(jnp.array([a1_a2_dphis.T,])).T[:,:,0],
            dist.MultivariateNormal(mu_evts, sigma_evts_neg).log_prob(jnp.array([a1_a2_neg_dphis.T,])).T[:,:,0]]), axis=0)
    
        log_wts += logp_normal
    
    else:
        sigma_evts = BW_matrices + jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)])) # (Nobs, 2, 2)
        sigma_sel = jnp.diag(jnp.array([jnp.square(sigma_spin), jnp.square(sigma_spin)]))
        mu_evts = jnp.array([mu_spin, mu_spin])
        
        logp_normal_sel = dist.MultivariateNormal(jnp.array([mu_spin, mu_spin]), sigma_sel).log_prob(a1_a2_sel.T)
        log_sel_wts += logp_normal_sel
    
        logp_normal = dist.MultivariateNormal(mu_evts, sigma_evts).log_prob(jnp.array([a1_a2s.T,])).T[:,:,0]
        log_wts += logp_normal
        
        
    # Adding the per event likelihood term
    log_like_max_l = jnp.max(log_wts, axis=1)
    delta_log_wts = log_wts - log_like_max_l[:, None]
    log_like = log_like_max_l + jnp.log(jnp.sum(jnp.exp(delta_log_wts), axis=1))
    numpyro.factor('log_likelihood', jnp.sum(log_like))
    
    # Selection effect term
    log_sel_max = jnp.max(log_sel_wts)
    delta_log_sel_wts = log_sel_wts - log_sel_max
    log_sel = log_sel_max + jnp.log(jnp.sum(jnp.exp(delta_log_sel_wts))) - jnp.log(Ndraw)
    numpyro.factor('selection', -Nobs * log_sel)
    
    # N eff cuts
    def log_smooth_neff_boundary(values, criteria):
        scaled_x = (values-criteria)/(0.05*criteria)
        return jax.lax.select(jnp.greater_equal(scaled_x, 0.0), 0.0, -jnp.power(scaled_x,10)) #jnp.log(jnp.greater(values, criteria)) #
    
    neff = jnp.exp(2*jnp.log(jnp.sum(jnp.exp(log_wts), axis=1)) - jnp.log(jnp.sum(jnp.exp(2*log_wts), axis=1)))
    min_neff = jnp.min(neff)
    numpyro.deterministic('neff', neff)
    numpyro.factor('neff_criteria', log_smooth_neff_boundary(min_neff, Nobs))
    
    log_mu2 = jnp.log(jnp.sum(jnp.exp(2*log_sel_wts))) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_sel - jnp.log(Ndraw) - log_mu2))
    neff_sel = jnp.exp(2*log_sel - log_s2)
    numpyro.deterministic('neff_sel', neff_sel)
    numpyro.factor('neff_sel_criteria', log_smooth_neff_boundary(neff_sel, 4*Nobs))