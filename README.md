# kaldi-diar-latte
This repo lists steps to perform diarization of audio with the kaldi toolkit. Diarization (who-spoken-when) is performed by decoding audio and generating transcriptions (speech-to-text). The transcriptions contain information on who was hypothesized (most likely) to be speaking. 

## 1. Install Kaldi toollkit. (http://kaldi-asr.org/)
## 2. Extract 40-mel filterbank (+ 3 pitch) features from audio, and normalize (CMVN - cepstral mean variance normalize).

``` 
nj=4 # number of jobs/cpus
steps/make_fbank_pitch.sh --nj $nj --cmd "run.pl" $test $test/log $test/data || exit 1;
steps/compute_cmvn_stats.sh $test $test/log $test/data || exit 1;
```

## 3. Decode audio utilizing the filterbank features and graph that contains lexicon, language model, and acoustic model combined.

```
nj=4 # number of jobs/cpus
steps/nnet/decode.sh --nj $nj \
        --cmd "run.pl" \
        --config conf/decode_dnn.config \
        --nnet $dir/4.nnet --acwt 0.1 \
        $tedliumDir/exp/tri3/graph $test $dir/decode_test-fhs_it4 || exit 1
```

## 4. Build your own language model.

```
ngram-count -text text.txt -lm text.txt.lm -kndiscount
```

## 5. Build your own lexicon.

Start with your list of words:
```
achilles
acid
acknowledge
acknowledgement
...
zebra
zoo
zoom
  ```

Generate pronunciations from this tool: http://www.speech.cs.cmu.edu/tools/lextool.html

## 6. Build your own acoustic model.
