# kaldi-diar-latte
This repo lists steps to perform diarization of audio with the kaldi toolkit. Diarization (who-spoken-when) is performed by decoding audio and generating transcriptions (speech-to-text). The transcriptions contain information on who was hypothesized (most likely) to be speaking. 

## 1. Install Kaldi toolkit 
From here: http://kaldi-asr.org/

If you are going to use kaldi with a GPU (to train DNN acoustic models for example), then make sure to install kaldi with `--use-cuda=yes (default)`. The following steps, except for building acoustic models will not require a GPU.

## 2. Extract features
Extract 40-mel filterbank (+ 3 pitch) features from audio, and normalize (CMVN - cepstral mean variance normalize).

``` 
nj=4 # number of jobs/cpus
test=data-fbank/test
steps/make_fbank_pitch.sh --nj $nj --cmd "run.pl" $test $test/log $test/data || exit 1;
steps/compute_cmvn_stats.sh $test $test/log $test/data || exit 1;
```

I worked with 8,000Hz single-channel .wav files. You can convert them like this.
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
NOTE2: The lextool will append numbers to words with multiple pronunications (`HELLO HH EH L OW; HELLO(1) HH AH L OW`), remove the number(s) `(1)` because it will not match the word(s) in your language model causing problems for kaldi to compile the information.

## 5. Build your own acoustic model.
I used one of Kaldi's standard recipes to train a DNN acoustic model. 
- Specifically the TEDLIUM s5 recipe: https://github.com/kaldi-asr/kaldi/tree/master/egs/tedlium/s5
-- The  TEDLIUM  corpus contains over 1,400 audio recordings and text transcription of TED talks, for a total of 120 hours  of  data  and  1.7M  words.
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
test=data-fbank/test
tedliumDir=$kaldi/egs/tedlium/s5
dir=$tedliumDir/exp/dnn4d-fbank_pretrain-dbn_dnn_smbr
steps/nnet/decode.sh --nj $nj \
        --cmd "run.pl" \
        --config conf/decode_dnn.config \
        --nnet $dir/4.nnet --acwt 0.1 \
        $tedliumDir/exp/tri3/graph $test $dir/decode_test-fhs_it4 || exit 1
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
The pipeline above was used in this paper: https://groups.csail.mit.edu/sls/publications/2018/Alhanai_NAACL18.pdf

