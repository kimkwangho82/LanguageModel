function wc(fname)
    chars=0
    words=0
    lines=0

    if(fname=="-") then it=io.lines()
    else it=io.lines(fname) end

    for line in it do
        lines=lines+1
        for word in string.gfind(line, "[^%s]+") do words=words+1 end
        chars=chars+string.len(line)+1
    end

    return chars, words, lines
end

--c, w, l = wc('/home/kwangho/research/torch_study/LanguageModel/data/wsj37000k.org.sentence.txt.upper.voc20k_unk.shuffle.head3700k')
--print(c, w, l)
