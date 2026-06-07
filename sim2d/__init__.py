"""2D pursuit-evasion sim for the Split Decision feint experiment.

Designed to be Isaac Sim action-contract compatible: the policy outputs
(v_cmd, omega_cmd) normalized to [-1, 1], the env scales by v_max/omega_max
and applies Ackermann coupling. Same contract works for JetBot/Carter in
Isaac Sim with only a wheel-mapping adapter.
"""
