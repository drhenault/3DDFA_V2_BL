vadMichal = readtable('../dumps/vad.csv');
vadMichal.vadDagcDecFinal(vadMichal.seconds > 10 & vadMichal.seconds < 16.8) = 0;
vadMichal.vadDagcDecFinal(vadMichal.seconds > 25.8) = 0;

vadMarek = readtable('../dumps/vad.csv');
vadMarek.vadDagcDecFinal(vadMarek.seconds < 10) = 0;

figure
subplot(3,1,1)
stairs(vadMichal.seconds, vadMichal.vadDagcDecFinal);
title('VAD Michal')
ylim([-0.1, 1.1])

subplot(3,1,2)
stairs(vadMarek.seconds, vadMarek.vadDagcDecFinal);
ylim([-0.1, 1.1])

title('VAD Marek')

subplot(3,1,3)
vad = readtable('../dumps/vad.csv');
stairs(vad.seconds, vad.vadDagcDecFinal);
title('VAD Global')
ylim([-0.1, 1.1])
