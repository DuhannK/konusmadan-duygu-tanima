## McNemar testi - tüm test kümesi
| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM-öznitelik | SVM-eGeMAPS | 49.0% | 50.5% | 218 | 239 | 0,349 | 0,699 | Hayır |
| SVM-öznitelik | SVM-w2v | 49.0% | 69.7% | 98 | 395 | < 0,001 | < 0,001 | Evet |
| SVM-öznitelik | MLP-w2v | 49.0% | 69.9% | 110 | 410 | < 0,001 | < 0,001 | Evet |
| SVM-öznitelik | CNN | 49.0% | 54.8% | 196 | 279 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | SVM-w2v | 50.5% | 69.7% | 107 | 383 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | MLP-w2v | 50.5% | 69.9% | 118 | 397 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | CNN | 50.5% | 54.8% | 207 | 269 | 0,005 | 0,016 | Evet |
| SVM-w2v | MLP-w2v | 69.7% | 69.9% | 77 | 80 | 0,873 | 0,873 | Hayır |
| SVM-w2v | CNN | 69.7% | 54.8% | 326 | 112 | < 0,001 | < 0,001 | Evet |
| MLP-w2v | CNN | 69.9% | 54.8% | 335 | 118 | < 0,001 | < 0,001 | Evet |

## McNemar testi - RAVDESS
| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM-öznitelik | SVM-eGeMAPS | 44.2% | 45.2% | 37 | 39 | 0,909 | 0,909 | Hayır |
| SVM-öznitelik | SVM-w2v | 44.2% | 78.8% | 8 | 80 | < 0,001 | < 0,001 | Evet |
| SVM-öznitelik | MLP-w2v | 44.2% | 80.8% | 9 | 85 | < 0,001 | < 0,001 | Evet |
| SVM-öznitelik | CNN | 44.2% | 52.9% | 34 | 52 | 0,067 | 0,267 | Hayır |
| SVM-eGeMAPS | SVM-w2v | 45.2% | 78.8% | 6 | 76 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | MLP-w2v | 45.2% | 80.8% | 9 | 83 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | CNN | 45.2% | 52.9% | 33 | 49 | 0,098 | 0,293 | Hayır |
| SVM-w2v | MLP-w2v | 78.8% | 80.8% | 6 | 10 | 0,454 | 0,909 | Hayır |
| SVM-w2v | CNN | 78.8% | 52.9% | 64 | 10 | < 0,001 | < 0,001 | Evet |
| MLP-w2v | CNN | 80.8% | 52.9% | 68 | 10 | < 0,001 | < 0,001 | Evet |

## McNemar testi - CREMA-D
| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM-öznitelik | SVM-eGeMAPS | 49.8% | 51.4% | 181 | 200 | 0,356 | 0,713 | Hayır |
| SVM-öznitelik | SVM-w2v | 49.8% | 68.2% | 90 | 315 | < 0,001 | < 0,001 | Evet |
| SVM-öznitelik | MLP-w2v | 49.8% | 68.1% | 101 | 325 | < 0,001 | < 0,001 | Evet |
| SVM-öznitelik | CNN | 49.8% | 55.1% | 162 | 227 | 0,001 | 0,005 | Evet |
| SVM-eGeMAPS | SVM-w2v | 51.4% | 68.2% | 101 | 307 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | MLP-w2v | 51.4% | 68.1% | 109 | 314 | < 0,001 | < 0,001 | Evet |
| SVM-eGeMAPS | CNN | 51.4% | 55.1% | 174 | 220 | 0,023 | 0,070 | Hayır |
| SVM-w2v | MLP-w2v | 68.2% | 68.1% | 71 | 70 | 1,000 | 1,000 | Hayır |
| SVM-w2v | CNN | 68.2% | 55.1% | 262 | 102 | < 0,001 | < 0,001 | Evet |
| MLP-w2v | CNN | 68.1% | 55.1% | 267 | 108 | < 0,001 | < 0,001 | Evet |
