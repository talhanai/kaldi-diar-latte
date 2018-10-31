# kaldi-diar-latte
This repo lists steps to perform diarization of audio with the kaldi toolkit. Diarization (who-spoken-when) is performed by decoding audio and generating transcriptions (speech-to-text). The transcriptions contain information on who was hypothesized (most likely) to be speaking. 

1. Install Kaldi toollkit. (http://kaldi-asr.org/)
2. Extract 40-mel filterbank (+ 3 pitch) features from audio, and normalize (CMVN - cepstral mean variance normalize).

``` 
nj=10 # number of jobs
steps/make_fbank_pitch.sh --nj $nj --cmd "run.pl" data data/log data/data || exit 1;
steps/compute_cmvn_stats.sh data data/log data/data || exit 1;
```

3. Decode audio utilizing the filterbank features.
4. Build your own language model.
5. Build your own acoustic model.
