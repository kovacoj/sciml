| Method | Offline cost | Online cost | Current quality |
|---|---|---|---|
| Reference IBM | none | ~28.0s | CFD |
| Reference U-Net | hundreds of full CFD + training | ~0.2s | excellent u, weaker p |
| Our W$_{20}$ distillation | 256×20 SIMPLE + training | 13.6ms | approximate state |
| Our NN + 5 SIMPLE | same | 0.49s | better finite-budget u |
| Full DAFoam | none | 60.0s | reference |
