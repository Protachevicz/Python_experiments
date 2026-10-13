reset
set terminal postscript eps size 14,7 enhanced color font 'Times, 42'
set output "atratores.eps"
#set terminal jpeg size 1280,920 enhanced font 'Times,22'
#set output "atratores.jpeg"
set palette model RGB
set multiplot layout 2,3
##################P1-3D##############################
reset
set xlabel 'e' font 'Times, 48'
set ylabel 'i' font 'Times, 48'
set zlabel 's' font 'Times, 48'
set xrange [-1:40]
set yrange [-1:40]
set zrange [100:600]
set ytics 20
set xtics 20
set ztics 300
set xlabel offset 0,-.7
set ylabel offset 0,-1
set zlabel offset 3,0
set ytics offset 0,-.2
set view 66,117,1,1
set xtics add ('0.02' 20, '0.04' 40)
set ytics add ('0.02' 20, '0.04' 40)
set ztics add ('0.6' 600, '0.3' 300, '0.1' 100)
splot "caos_p1_alfa130.88200.dat"  u 2:3:1 ls 7 pointsize 0.6 lt rgb 'red' notitle, 'per_p1_alfa130.88200.dat' u 2:3:1 ls 7 pointsize 2.5 lt rgb "#007cff" notitle
##################P2-3D##############################
reset
set xlabel 'e' font 'Times, 48'
set ylabel 'i' font 'Times, 48'
set zlabel 's' font 'Times, 48'
set xrange [-1:30]
set yrange [-1:40]
set zrange [100:600]
set ytics 20
set xtics 15
set ztics 300
set ztics add ('100' 100)
set xlabel offset 1,-1
set ylabel offset 0,-1
set zlabel offset 3,0
set view 49,115,1,1
set xtics add ('0.15' 15, '0.03' 30)
set ytics add ('0.02' 20, '0.04' 40)
set ztics add ('0.6' 600, '0.3' 300, '0.1' 100)
splot "caos_p2_alfa161.70276.dat"  u 2:3:1 ls 7 pointsize 0.6 lt rgb "#007cff" notitle, 'per_p2_alfa161.70276.dat' u 2:3:1 ls 7 pointsize 2.5 notitle
##################P3-3D##############################
reset
set xlabel 'e' font 'Times, 48'
set ylabel 'i' font 'Times, 48'
set zlabel 's' font 'Times, 48'
set xrange [-1:20]
set yrange [-1:60]
set zrange [100:600]
set ytics 30
set xtics 10
set ztics 300
set ztics add ('100' 100)
set xlabel offset 0,-.7
set ylabel offset -.5,0
set zlabel offset 3,0
set xtics offset 0,-.2
set view 63,168,1,1
set ztics add ('0.6' 600, '0.3' 300, '0.1' 100)
set ytics ('0.03' 30, '0.06' 60)
splot "caos_p3_alfa303.884116.dat"  u 2:3:1 ls 7 pointsize 0.6 lt rgb 'red' notitle
#####################################################
##################P1-2D##############################
reset
set xlabel 's' font 'Times, 48'
set ylabel 'i' font 'Times, 48'
set xrange [100:600]
set yrange [-1:60]
set ytics 20
set xtics 300
set xtics add ('0.1' 100, '0.3' 300, '0.6' 600)
plot "caos_p1_alfa130.88200.dat"  u 1:3 ls 7 pointsize 0.6 lt rgb 'red' notitle, 'per_p1_alfa130.88200.dat' u 1:3 ls 7 pointsize 2.5 lt rgb "#007cff" notitle
##################P2-2D##############################
reset
set xlabel 's' font 'Times, 48'
set ylabel 'i' font 'Times, 48'
set xrange [100:600]
set yrange [-1:60]
set ytics 20
set xtics 300
set xtics add ('100' 100)
plot "caos_p2_alfa161.70276.dat"  u 1:3 ls 7 pointsize 0.6 lt rgb "#007cff" notitle, 'per_p2_alfa161.70276.dat' u 1:3 ls 7 pointsize 2.5 notitle
##################P3-2D##############################
reset
set xlabel 's' font 'Times, 48'
set ylabel 'i' font 'Times, 48'
set xrange [100:600]
set yrange [-2:60]
set ytics 30
set xtics 300
set xtics add ('100' 100)
plot "caos_p3_alfa303.884116.dat"  u 1:3 ls 7 pointsize 0.6 lt rgb 'red' notitle
#######
unset multiplot
set output
