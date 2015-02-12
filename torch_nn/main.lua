require 'train'
require 'predict_next_word'

---------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Language Modeling ')
cmd:text('Example:')
cmd:text('$> th main.lua --train train.txt --valid valid.txt --test test.txt --vocab vocab.txt --n 3')
cmd:text('Options:')
cmd:option('--train', 'train.txt', 'training file')
cmd:option('--valid', 'valid.txt', 'validation file')
cmd:option('--test', 'test.txt', 'test file')
cmd:option('--vocab', 'vocab.txt', 'vocabulary file')
cmd:option('--n', 3, '"n" for n-gram')
cmd:text()

opt = cmd:parse(arg or {})
print(opt)

------------------------------------------------

use_manual_technique = false;
epochs = 1;

-- Manual Training seems to require more epochs to get a similar error rate.
if use_manual_technique == true then epochs = 3; end

model = train(epochs,use_manual_technique);
torch.save("train.model", model)
--model = torch.load("train.model")
--predict_next_word('who', 'are', 'you', model, 8);
predict_next_word('WHO', 'ARE', 'YOU', model, 8);
--predict_next_word('do', 'you', 'know', model, 250);
