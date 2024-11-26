import pstats

p = pstats.Stats("game_profile.prof")
p.strip_dirs().sort_stats("time").print_stats(10) # Sort by time and show top 10 functions