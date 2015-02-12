#!/usr/bin/env th

require 'nn'
_ = require 'underscore'
require 'util/wc'

-- 1. ****** --

cmd = torch.CmdLine()
cmd:text()
cmd:text('Language Modeling ')
cmd:text('Example:')
cmd:text('$> th data_t7.lua --train train.txt --valid valid.txt --test test.txt --vocab vocab.txt')
cmd:text('Options:')
cmd:option('--train', 'train.txt', 'training file')
cmd:option('--valid', 'valid.txt', 'validation file')
cmd:option('--test', 'test.txt', 'test file')
cmd:option('--vocab', 'vocab.txt', 'vocabulary file')
cmd:text()

opt = cmd:parse(arg or {})
print(opt)

-----------------------------------------------------------

sentence_start = "<S>"
sentence_end = "</S>"
unknown_word = "<UNK>"


function load_vocab(vocab_file)
    print("vocab_file : " .. vocab_file)
    
    vocab ={}
    vocab_ByIndex = {}
    vocab_size = 0
    
    f_vocab = io.open(vocab_file, "r")

    idx_voc = 1
    for line in io.lines(vocab_file) do
        word = line
        vocab[word] = idx_voc
        table.insert(vocab_ByIndex, word)
        idx_voc = idx_voc + 1
    end

    f_vocab:close()
    
    vocab[sentence_start] = idx_voc
    idx_voc = idx_voc + 1
    table.insert(vocab_ByIndex, sentence_start)

    vocab[sentence_end] = idx_voc
    idx_voc = idx_voc + 1
    table.insert(vocab_ByIndex, sentence_end)

    vocab_size = (idx_voc - 1)

    return vocab, vocab_ByIndex, vocab_size
end
-----------------------------------------------------------

function load_data(data_file, vocab)
    print("data_file : " .. data_file)
    
    n_chars, n_words, n_lines = wc(data_file)

    res_int_storage = torch.IntStorage(2*(n_words+n_lines)):fill(0)
    
    f_data = io.open(data_file, "r")

    idx_word = 1
    idx_storage = 1
    for line in io.lines(data_file) do
        --print(line)
        idx_sent = idx_word
        for word in string.gmatch(line, "[^%s]+") do
            --print(word)
            --table.insert(sent, word)
            res_int_storage[idx_storage] = idx_sent
            idx_storage = idx_storage + 1
            res_int_storage[idx_storage] = vocab[word]
            idx_storage = idx_storage + 1

            idx_word = idx_word + 1
        end
        res_int_storage[idx_storage] = idx_sent
        idx_storage = idx_storage + 1
        res_int_storage[idx_storage] = vocab[sentence_end]
        idx_storage = idx_storage + 1

        idx_word = idx_word + 1
    end

    f_data:close()

    res_int_tensor = torch.IntTensor(res_int_storage):resize((n_words+n_lines),2)

    return res_int_tensor
end	
------------------------------------------------------------
function freq_word(data_file, vocab_size)
    print("data_file : " .. data_file)
   
    words_freq = {}
    
    print("vocab_size : ", vocab_size)
    for i=1,vocab_size do
        table.insert(words_freq, 0)
    end
    --print(words_freq)

    f_data = io.open(data_file, "r")

    for line in io.lines(data_file) do
        --print(line)
        --print(sentence_start, vocab[sentence_start])
        pre_freq = words_freq[vocab[sentence_start]]
        --print("pre_freq : ", pre_freq)
        words_freq[vocab[sentence_start]] = pre_freq + 1

        for word in string.gmatch(line, "[^%s]+") do
            --print(word)
            --table.insert(sent, word)
            pre_freq = words_freq[vocab[word]]
            words_freq[vocab[word]] = pre_freq + 1
        end
        pre_freq = words_freq[vocab[sentence_end]]
        words_freq[vocab[sentence_end]] = pre_freq + 1
    end
    
    return torch.IntTensor(words_freq)
end 

-------------------------------------------------

vocab, vocab_ByIndex, vocab_size = load_vocab(opt.vocab)
trainData = load_data(opt.train, vocab)
validData = load_data(opt.valid, vocab)
testData = load_data(opt.test, vocab)

wordFreq = freq_word(opt.train, vocab_size)
--print(wordFreq)
--print(trainData)
--print(validData)
--print(testData)
--print(vocab)
--print(vocab_ByIndex)

torch.save("train_data.th7", trainData)
torch.save("valid_data.th7", validData)
torch.save("test_data.th7", testData)

torch.save("word_map.th7", vocab_ByIndex)
torch.save("word_freq.th7", wordFreq)
