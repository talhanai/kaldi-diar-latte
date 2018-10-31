#!/bin/bash

source cmd.sh
source path.sh

src_dir=$1
lm=$2
lexicon=$3
output=$4

# lm_srcdir=current/lang/4gram-mincount
# lm_srcdir=current/lang/3gram
# lang=current/lang

#   Command taken from WSJ recipe
gunzip -c $lm | \
  arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$src_dir/words.txt \
      --osymbols=$src_dir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > $output/G.fst || exit 1;
  fstisstochastic $output/G.fst


  # Everything below is only for diagnostic.
  # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
  # this might cause determinization failure of CLG.
  # #0 is treated as an empty word.

# lexicon=current/local/dict/lexicon.txt
# test=$lang
test=$output
tmpdir=/tmp
mkdir -p $tmpdir/g
awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/g/select_empty.fst.txt | \
    fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/g/empty_words.fst
fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' && 
    echo "Language model has cycles with empty words" && exit 1
rm -r $tmpdir/g




# eof
