th lm.lua --train train.txt -- vocab vocab.txt

th lm.lua --train ./data/wsj37000k.org.sentence.txt.upper.voc20k_unk.shuffle.head3700k --valid ./data/wsj37000k.org.sentence.txt.upper.voc20k_unk.shuffle.tail_16255sent --test ./data/wsj37000k.org.sentence.txt.upper.voc20k_unk.shuffle.tail_16255sent --vocab ./data/wsj_20k.vocab --n 3
