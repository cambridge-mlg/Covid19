using Turing, DrWatson

# Allows us to use `safesave(filename, chain)` to ensure that we do not overwrite any chains
DrWatson._wsave(filename, chain::Chains) = write(filename, chain)
