run_path="/home/alta/BLTSpeaking/exp-pr450/jacobian_investigation/scripts/run_train.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/jacobian_investigation/logs/result_log"

log_suffix=$1

# 1. Jacobian:  nj, j, jnp
# 2. Reweigher: nrw, rw, rwoff
# 3. Test prop: 0.2, 0.3, 0.4, 0.5, 0.6

$run_path "${txt_base}_01${log_suffix}" nj nrw 0.2
$run_path "${txt_base}_02${log_suffix}" nj nrw 0.3
$run_path "${txt_base}_03${log_suffix}" nj nrw 0.4
$run_path "${txt_base}_04${log_suffix}" nj nrw 0.5
$run_path "${txt_base}_05${log_suffix}" nj nrw 0.6

$run_path "${txt_base}_11${log_suffix}" j nrw 0.2
$run_path "${txt_base}_12${log_suffix}" j nrw 0.3
$run_path "${txt_base}_13${log_suffix}" j nrw 0.4
$run_path "${txt_base}_14${log_suffix}" j nrw 0.5
$run_path "${txt_base}_15${log_suffix}" j nrw 0.6

$run_path "${txt_base}_21${log_suffix}" jnp nrw 0.2
$run_path "${txt_base}_22${log_suffix}" jnp nrw 0.3
$run_path "${txt_base}_23${log_suffix}" jnp nrw 0.4
$run_path "${txt_base}_24${log_suffix}" jnp nrw 0.5
$run_path "${txt_base}_25${log_suffix}" jnp nrw 0.6


$run_path "${txt_base}_31${log_suffix}" nj rw 0.2
$run_path "${txt_base}_32${log_suffix}" nj rw 0.3
$run_path "${txt_base}_33${log_suffix}" nj rw 0.4
$run_path "${txt_base}_34${log_suffix}" nj rw 0.5
$run_path "${txt_base}_35${log_suffix}" nj rw 0.6

$run_path "${txt_base}_41${log_suffix}" j rw 0.2
$run_path "${txt_base}_42${log_suffix}" j rw 0.3
$run_path "${txt_base}_43${log_suffix}" j rw 0.4
$run_path "${txt_base}_44${log_suffix}" j rw 0.5
$run_path "${txt_base}_45${log_suffix}" j rw 0.6

$run_path "${txt_base}_51${log_suffix}" jnp rw 0.2
$run_path "${txt_base}_52${log_suffix}" jnp rw 0.3
$run_path "${txt_base}_53${log_suffix}" jnp rw 0.4
$run_path "${txt_base}_54${log_suffix}" jnp rw 0.5
$run_path "${txt_base}_55${log_suffix}" jnp rw 0.6


$run_path "${txt_base}_61${log_suffix}" nj rwoff 0.2
$run_path "${txt_base}_62${log_suffix}" nj rwoff 0.3
$run_path "${txt_base}_63${log_suffix}" nj rwoff 0.4
$run_path "${txt_base}_64${log_suffix}" nj rwoff 0.5
$run_path "${txt_base}_65${log_suffix}" nj rwoff 0.6

$run_path "${txt_base}_71${log_suffix}" j rwoff 0.2
$run_path "${txt_base}_72${log_suffix}" j rwoff 0.3
$run_path "${txt_base}_73${log_suffix}" j rwoff 0.4
$run_path "${txt_base}_74${log_suffix}" j rwoff 0.5
$run_path "${txt_base}_75${log_suffix}" j rwoff 0.6

$run_path "${txt_base}_81${log_suffix}" jnp rwoff 0.2
$run_path "${txt_base}_82${log_suffix}" jnp rwoff 0.3
$run_path "${txt_base}_83${log_suffix}" jnp rwoff 0.4
$run_path "${txt_base}_84${log_suffix}" jnp rwoff 0.5
$run_path "${txt_base}_85${log_suffix}" jnp rwoff 0.6

