Bregma_ml = 15.1
Bregma_pa = 50.5


AM_BC = [Bregma_ml + 4.0, Bregma_pa - 0.26]
AL_BC = [Bregma_ml + 6.8, Bregma_pa - 0.26]
PM_BC = [Bregma_ml + 4.0, Bregma_pa - 4.16]
PL_BC = [Bregma_ml + 7.4, Bregma_pa - 4.16]

AM_BS = [Bregma_ml - 2.4, Bregma_pa - 11.8]
AL_BS = [Bregma_ml - 3.2, Bregma_pa - 11.8]
PM_BS = [Bregma_ml - 1.9, Bregma_pa - 13.3]
PL_BS = [Bregma_ml - 3.2, Bregma_pa - 13.3]

print(f'Barrel cortex:\nAM = {AM_BC}\nAL = {AL_BC}\nPM = {PM_BC}\nPL = {PL_BC}\n\n'
      f'Brainstem nuclei:\nAM = {AM_BS}\nAL = {AL_BS}\nPM = {PM_BS}\nPL = {PL_BS}')