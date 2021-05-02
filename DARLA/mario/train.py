from DARLA.mario.dae.dae import DAE
from DARLA.mario.beta_vae.beta_vae import BetaVAE

dae = DAE()
dae.train()
vae = BetaVAE(dae, 0.1)
vae.train()