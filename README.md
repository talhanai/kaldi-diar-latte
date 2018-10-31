# kaldi-diar-latte
This repo lists steps to perform text-based diarization of audio with the kaldi toolkit. Diarization (who-spoken-when) is performed by decoding audio and generating transcriptions (speech-to-text). The transcriptions contain information on who was hypothesized (most likely) to be speaking, and what they were likely to be saying. 

## 1. Install Kaldi toolkit 
From here: http://kaldi-asr.org/

If you are going to use kaldi with a GPU (to train DNN acoustic models for example), then make sure to install kaldi with `--use-cuda=yes (default)`. The following steps, except for building acoustic models will not require a GPU.

## 2. Extract features
Extract 40-mel filterbank (+ 3 pitch) features from audio, and normalize (CMVN - cepstral mean variance normalize).

``` 
nj=4 # number of jobs/cpus
data=data-fbank/test
steps/make_fbank_pitch.sh --nj $nj --cmd "run.pl" $data $data/log $data/data || exit 1;
steps/compute_cmvn_stats.sh $data $data/log $data/data || exit 1;
```

The **$test** directory will containing the files `wav.scp`, `text`, `utt2spk`, and `spk2utt` that you must generate in a (kaldi pre-defined) format. Details on these files can be [found here](http://kaldi-asr.org/doc/data_prep.html). They should look like this:
```
$> head *

==> spk2utt <==
SID-0001 SID-0001
SID-0002 SID-0002
SID-0003 SID-0003

==> text <==
SID-0001 <empty>
SID-0002 <empty>
SID-0003 <empty>

==> utt2spk <==
SID-0001 SID-0001
SID-0002 SID-0002
SID-0003 SID-0003

==> wav.scp <==
SID-0001 DVR_226ABCDEF_DATEXY_TID1.wav
SID-0002 DVR_226ABCDEF_DATEXY_TID2.wav
SID-0003 DVR_226ABCDEF_DATEXY_TID3.wav

```

Once you generate features using the `steps/make_fbank_pitch.sh` and `steps/compute_cmvn_stats.sh` you will have the following files in `data-fbank/test`: `feats.scp` and `cmvn.scp`. These files are pointers to the location of the features (`*.ark` files)

```
$> head *

==> cmvn.scp <==
SID-0001 data-fbank/test/data/cmvn_test7501.ark:9
SID-0002 data-fbank/test/data/cmvn_test7501.ark:737
SID-0003 data-fbank/test/data/cmvn_test7501.ark:1465

==> feats.scp <==
SID-0001 data-fbank/test/data/raw_fbank_pitch_fhs_fbank40_pitch.1.ark:9
SID-0002 data-fbank/test/data/raw_fbank_pitch_fhs_fbank40_pitch.1.ark:28162416
SID-0003 data-fbank/test/raw_fbank_pitch_fhs_fbank40_pitch.1.ark:58370247

```

These features will be contained in `data-fbank/test/data/*`. You can take a look at the feature values like this:

```
$> copy-feats ark:data-fbank/test/data/cmvn_test7501.ark  ark,t:- | head
SID-0001  [
  1.00398e+07 9766473 9383610 1.005838e+07 1.022724e+07 1.014023e+07 9981066 1.01312e+07 1.019235e+07 1.049552e+07 1.061847e+07 1.073239e+07 1.074094e+07 1.077999e+07 1.079588e+07 1.079876e+07 1.082031e+07 1.081672e+07 1.081716e+07 1.082457e+07 1.091485e+07 1.105305e+07 1.120228e+07 1.122947e+07 1.137938e+07 1.151344e+07 1.15209e+07 1.140444e+07 1.14364e+07 1.149396e+07 1.148206e+07 1.146546e+07 1.148979e+07 1.160972e+07 1.165308e+07 1.180215e+07 1.189667e+07 1.188064e+07 1.176459e+07 1.142168e+07 -262316 60380.89 -128.0268 654931 
  1.540788e+08 1.459334e+08 1.353966e+08 1.557601e+08 1.614248e+08 1.592616e+08 1.545848e+08 1.595599e+08 1.620591e+08 1.723894e+08 1.767884e+08 1.805394e+08 1.808433e+08 1.820892e+08 1.825487e+08 1.825446e+08 1.831315e+08 1.829549e+08 1.827662e+08 1.832903e+08 1.862949e+08 1.907063e+08 1.956348e+08 1.967739e+08 2.019863e+08 2.065044e+08 2.069619e+08 2.034571e+08 2.047451e+08 2.066113e+08 2.062672e+08 2.060059e+08 2.067537e+08 2.105213e+08 2.122883e+08 2.175693e+08 2.21419e+08 2.210206e+08 2.165605e+08 2.044845e+08 133956.9 195603.2 35981.67 0 ]
SID-0002  [
  8384802 8486188 8833670 9253396 1.001631e+07 1.177026e+07 1.189225e+07 1.097206e+07 1.056413e+07 1.094213e+07 1.100873e+07 1.126306e+07 1.110211e+07 1.096557e+07 1.086174e+07 1.084818e+07 1.076636e+07 1.08146e+07 1.086717e+07 1.100974e+07 1.123237e+07 1.140382e+07 1.136161e+07 1.150994e+07 1.171923e+07 1.158355e+07 1.14348e+07 1.17521e+07 1.183192e+07 1.185442e+07 1.186898e+07 1.18521e+07 1.193353e+07 1.21977e+07 1.227238e+07 1.225209e+07 1.228296e+07 1.222558e+07 1.230039e+07 1.193534e+07 -252066.3 9694.388 -165.0663 702499 
  1.026882e+08 1.045965e+08 1.135667e+08 1.247654e+08 1.44898e+08 1.98017e+08 2.019952e+08 1.726382e+08 1.617604e+08 1.738974e+08 1.761198e+08 1.837795e+08 1.790269e+08 1.751063e+08 1.719396e+08 1.713151e+08 1.686291e+08 1.69887e+08 1.7119e+08 1.75409e+08 1.822639e+08 1.880301e+08 1.8651e+08 1.911555e+08 1.981422e+08 1.937729e+08 1.893091e+08 1.99298e+08 2.020286e+08 2.028399e+08 2.034605e+08 2.02963e+08 2.057297e+08 2.144382e+08 2.171906e+08 2.169324e+08 2.180554e+08 2.162135e+08 2.186301e+08 2.064577e+08 115918.3 82817.53 20336.01 0 ]
SID-0003  [
  4230204 5064260 5888814 6212606 6326862 6337283 6163856 6327585 6323948 6330220 6211693 6182526 6147778 6071172 5987304 5898564 5775111 5718952 5675966 5664286 5628709 5613251 5632297 5660430 5706630 5725274 5726684 5675726 5606198 5616602 5640256 5680830 5716024 5709345 5693758 5701922 5813593 5905173 5787294 5310092 -95720.46 -11746.92 -145.6066 463627 
  3.9893e+07 5.648841e+07 7.6305e+07 8.547831e+07 8.91432e+07 8.921225e+07 8.389834e+07 8.88213e+07 8.971312e+07 9.034718e+07 8.714226e+07 8.691971e+07 8.618955e+07 8.375963e+07 8.110999e+07 7.842232e+07 7.508191e+07 7.368099e+07 7.272582e+07 7.257e+07 7.177961e+07 7.133672e+07 7.17921e+07 7.244643e+07 7.347638e+07 7.375366e+07 7.358233e+07 7.211532e+07 7.030644e+07 7.03931e+07 7.091698e+07 7.186106e+07 7.260594e+07 7.22487e+07 7.167794e+07 7.175174e+07 7.445702e+07 7.672863e+07 7.36327e+07 6.211307e+07 53783.92 38258.19 7460.848 0 ]
SID-0004  [

```


NOTE: I worked with 8,000Hz single-channel .wav files. You can convert them like this.
```
$> ffmpeg -n -i $inputfilename -ar 8000 -ac 1 $outputfilename.wav 

    or

$> sox $inputfilename $outputfilename.wav channels 1 rate 8k
```

## 3. Build your language model.

You might want to utilize your own text to build a language model (i.e. pattern of language word sequences).

- Install the SRILM toolkit: http://www.speech.sri.com/projects/srilm/download.html
- Run the following command to generate your language model that kaldi can later use in its decoder (it is a tri-gram model with Knesser-Ney discounting).

```
$> ngram-count -text text.txt -lm text.txt.lm.gz -kndiscount
```

## 4. Build your own lexicon.

1. Extract your list of words (the same words that is in your `text.txt` used in your language model):
```
$> sed 's/ /\/g text.txt | sed '/^$/d' | sort | uniq > vocab.txt # prints your vocabulary to file
$> cat vocab.txt # take a look at the list of words
achilles
acid
acknowledge
acknowledgement
...
zebra
zoo
zoom
  ```

2. Generate pronunciations from this tool: http://www.speech.cs.cmu.edu/tools/lextool.html

The results will look something like this, which will be your `lexicon.txt`:
```
abduct AE B D AH K T
abducted AE B D AH K T IH D
abducted AH B D AH K T IH D
```

NOTE: Make sure the list of words match what is contained in the text of the language model, otherwise Kaldi will complain when it combines the data. It can't understand that there are words in the language model that don't have pronunciations.

NOTE2: The lextool will append numbers to words with multiple pronunications (`hello HH EH L OW; hello(1) HH AH L OW`), remove the number(s) `(1)` because it will not match the word(s) in your language model causing problems for kaldi to compile the information. It will look like this: `hello HH EH L OW; hello HH AH L OW`

## 5. Build your own acoustic model.
I used one of Kaldi's standard recipes to train a DNN acoustic model. 
- Specifically the TEDLIUM s5 recipe: https://github.com/kaldi-asr/kaldi/tree/master/egs/tedlium/s5
    - The  TEDLIUM  corpus contains over 1,400 audio recordings and text transcription of TED talks, for a total of 120 hours  of  data  and  1.7M  words.
- Make sure to run `run.sh` all the way upto and including `local/nnet/run_dnn.sh`
- NOTE: My experiments were with audio sampled at 8,000Hz, the tedlium corpus files are 16,000Hz so I downsampled them first before building the acoustic model (with `run.sh`).

## 6. Combine data
 During the acoustic model training, lexicon and language models were generated on the tedlium corpus. (You can try decoding with it but it will likely transcribe the audio poorly). So this is how you can combine your own lexical and language model.
 
 ```
 utils/prepare_lang.sh $dict "<unk>" $lang $lang
 preprocess/format_lm.sh $lang $lang/text.txt.lm.gz $dict/lexicon.txt $lang
 utils/mkgraph.sh $lang $exp $exp/graph
 ```

## 7. Decode audio 
Decode audio utilizing the filterbank features and graph that contains lexicon, language model, and acoustic model combined.

```
nj=4 # number of jobs/cpus
data=data-fbank/test
tedliumDir=$kaldi/egs/tedlium/s5
dir=$tedliumDir/exp/dnn4d-fbank_pretrain-dbn_dnn_smbr
steps/nnet/decode.sh --nj $nj \
        --cmd "run.pl" \
        --config conf/decode_dnn.config \
        --nnet $dir/4.nnet --acwt 0.1 \
        $tedliumDir/exp/tri3/graph $data $dir/decode_test-fhs_it4 || exit 1
```

# Reference
```
@inproceedings{al2018role,
  title={Role-specific Language Models for Processing Recorded Neuropsychological Exams},
  author={Al Hanai, Tuka and Au, Rhoda and Glass, James},
  booktitle={Proceedings of the 2018 
  Conference of the North American Chapter of the Association for Computational Linguistics: 
  Human Language Technologies, Volume 2 (Short Papers)}, 
  volume={2},
  pages={746--752},
  year={2018}
}
```
The pipeline above was used in [this paper](https://groups.csail.mit.edu/sls/publications/2018/Alhanai_NAACL18.pdf)

DISCLAIMER: The user accepts the code / configuration / repo AS IS, WITH ALL FAULTS.

