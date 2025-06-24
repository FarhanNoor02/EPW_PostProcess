set terminal pngcairo size 1200,900 enhanced font "Times,25"
set output "ncn_aniso_del.png"

set ylabel "{/=30 {Δ_{nk} (meV)}}" offset 1, 0
set xlabel "{/=30 {Temperature (K)}}" offset 0, 0.5

set lmargin screen 0.18
set rmargin screen 0.96
set tmargin screen 0.94
set bmargin screen 0.20

set tics font ",30"
set xtics font ",20" rotate by -45
set ytics font ",20"

set xtics 1
set mxtics 1
set ytics 0.5
set mytics 5

set key font ",20"
set key right top

set xrange [2.5:11.5]
set yrange [0:3]

scale = 0.4
set style fill solid
set style fill noborder

# Add vertical guide arrows at each integer temperature
do for [t in "3 4 5 6 7 8 9 10 11"] {
    set arrow from t, 0 to t, 2.8 nohead
}

# Label for mu
#set label 1 "μ_c^* = 0.05" font ",25"
#set label 1 at graph 0.75, 0.9

# Plot using boxxy style
plot for [temp=3:11] \
    sprintf('ncn.imag_aniso_gap0_%06.2f', temp) using \
    (($1 - temp) * scale / 2 + temp):($2 - 0.015):(($1 - temp) * scale / 2):(0.015) \
    with boxxy lc 0 notitle

