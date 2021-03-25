## TODO LIST

- [x] Convert audio to MFCC features using librosa
- [x] Debug CQCC/MFCC Code (matlab translation)
- [ ] Run CQCC code to get CQCC features
- [ ] feature extraction for PA database
- [ ] train single model for LA + PA data and train seperate models for both dataset.
- [x] GMM model on MFCC
- [x] SVM model on MFCC
- [ ] NNET model on MFCC
- [ ] GMM model on CQCC
- [ ] SVM model on CQCC
- [ ] NNET model on CQCC
- [x] Debug t-DCF code
- [x] calculate metrics, tDCF & EER
- [x] Basic Error Analysis


## Possible Audio Features

- [ ] MFCC
- [ ] linear & gemoetric CQCC
- [ ] Log-magnitude STFT
- [ ] x-vectors of size 512 as an input with kladi
- [ ] 60-dimensional LFCCs The frame size is 20ms and the hop size is 10ms
- [ ] 257-dimension spectrograms using 512-point FFT, and Kaldi toolkit to apply cepstral mean and variance normalization (cmvn)
- [ ] Blackman analysis window of 25 ms length with 10 ms of frame shift. Log magnitude spectrogram features (STFT) with 256 frequency bins
- [ ] Narrow Band Spectrum
- [ ] Full Band Spectrum
- [ ] Narrow & Full Band Spectrum 
- [ ] pre-detection as a preliminary step checks whether the input speech signal has zero temporal energy values
- [ ] MFPC coefficients
- [ ] phase spectrum, obtained by the Fourier Transform 
- [ ] wavelet-packet transform adapted to the mel scale
- [ ] all-pole group delay function (APGDF)
- [ ] fundamental frequency variation (FFV) to extract pitch variation at frame-level, provides complementary information to CQCC and APGDF.
- [ ] MGDF, CosPhase, RPS
- [ ] logspec
- [ ] inverted mel frequency cepstral coefficients (IMFCCs)
- [ ]  sub-band centroid magnitude coefficients (SCMC)
- [ ] longterm-average-spectrum (LTAS) feature




