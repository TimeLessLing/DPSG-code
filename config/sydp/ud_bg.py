import re
import json
from transformers.data import metrics
from tqdm import tqdm

split_token = '<fun_spt>'


special_tokens = [split_token, '[pid_token]', '[dep_token]', '[acl:cleft]', '[acl:relcl]', '[acl]', '[advcl:cleft]', '[advcl:tcl]', '[advcl]', '[advmod:emph]', '[advmod:tmod]', '[advmod]', '[amod]', '[appos]', '[aux:caus]', '[aux:pass]', '[aux]', '[case]', '[cc:preconj]', '[cc]', '[ccomp:pmod]', '[ccomp]', '[compound:prt]', '[compound]', '[conj]', '[cop]', '[csubj:pass]', '[csubj]', '[dep]', '[det:numgov]', '[det:nummod]', '[det:poss]', '[det:predet]', '[det]', '[discourse]', '[dislocated]', '[expl:impers]', '[expl:pass]', '[expl:poss]', '[expl:pv]', '[expl]', '[fixed]', '[flat:foreign]', '[flat:name]', '[flat]', '[goeswith]', '[iobj:agent]', '[iobj]', '[list]', '[mark]', '[nmod:agent]', '[nmod:npmod]', '[nmod:pmod]', '[nmod:poss]', '[nmod:range]', '[nmod:tmod]', '[nmod]', '[nsubj:caus]', '[nsubj:pass]', '[nsubj]', '[nummod:entity]', '[nummod:gov]', '[nummod]', '[obj:agent]', '[obj]', '[obl:agent]', '[obl:arg]', '[obl:mod]', '[obl:npmod]', '[obl:tmod]', '[obl]', '[orphan]', '[parataxis]', '[punct]', '[reparandum]', '[root]', '[vocative]', '[xcomp]', '[ADJ]', '[ADP]', '[ADV]', '[AUX]', '[CCONJ]', '[DET]', '[INTJ]', '[NOUN]', '[NUM]', '[PART]', '[PRON]', '[PROPN]', '[PUNCT]', '[SCONJ]', '[SYM]', '[VERB]', '[X]']

origin_tokens = [['adjective'], ['adposition'], ['adverb'], ['auxiliary'], ['coordinating', 'conjunction'], ['determiner'], ['interjection'], ['noun'], ['numeral'], ['particle'], ['pronoun'], ['proper', 'noun'], ['punctuation'], ['subordinating', 'conjunction'], ['symbol'], ['verb'], ['other'], 
    ['clefted', 'phrase', 'clause', 'modifier'], ['relative', 'clause', 'modifier'], ['clausal', 'modifier', 'of', 'noun'], ['clefted', 'adverbial', 'clause', 'modifier'], ['temporal', 'adverbial', 'clause'], ['adverbial', 'clause', 'modifier'], ['adverbial', 'emphasizing', 'word', 'intensifier'], ['temporal', 'adverbial', 'modifier'], ['adverbial', 'modifier'], ['adjectival', 'modifier'], ['appositional', 'modifier'], ['causative', 'auxiliary'], ['passive', 'auxiliary'], ['auxliary'], ['case', 'marking'], ['coordinating', 'conjunction', 'preconjunct'], ['coordinating', 'conjunction'], ['clausal', 'complement', 'phrase', 'modifier'], ['clausal', 'complement'], ['compound', 'phrasal', 'verb', 'particle'], ['compound'], ['conjunct'], ['copula'], ['clausal', 'passive', 'subject'], ['clausal', 'subject'], ['unspecified', 'dependency'], ['determiner', 'pronominal', 'quantifier', 'governing', 'the', 'case', 'of', 'the', 'noun'], ['determiner', 'pronominal', 'quantifier', 'agreeing', 'in', 'case', 'with', 'the', 'noun'], ['possessive', 'determiner'], ['predeterminer'], ['determiner'], ['discourse', 'element'], ['dislocated', 'elements'], ['impersonal', 'expletive'], ['reflexive', 'pronoun', 'used', 'in', 'relfexive', 'passive'], ['possessive', 'expletive'], ['reflexive', 'clitic', 'with', 'an', 'inherently', 'reflexive', 'verb'], ['expletive'], ['fixed', 'multiword', 'expression'], ['foreign', 'words'], ['flat', 'names'], ['flat', 'multiword', 'expression'], ['goes', 'with'], ['indirect', 'object'], ['indirect', 'agent', 'object'], ['list'], ['marker'],['agent', 'nominal', 'modifier'], ['noun', 'phrase', 'as', 'adverbial', 'nominal', 'modifier'], ['phrase', 'as', 'adverbial', 'nominal', 'modifier'], ['possessive', 'nominal', 'modifier'], ['range', 'nominal', 'modifier'], ['temporal', 'nominal', 'modifier'], ['nominal', 'modifier'], ['causative', 'nominal', 'subject'], ['passive', 'nominal', 'subject'], ['nominal', 'subject'], ['numeric', 'modifier', 'entity'], ['numeric', 'modifier', 'governing', 'the', 'case', 'of', 'the', 'noun'], ['numeric', 'modifier'], ['agent', 'object'], ['object'], ['oblique', 'agent', 'modifier'], ['oblique', 'argument'], ['oblique', 'modifier'], ['oblique', 'noun', 'phrase', 'as', 'adverbial', 'modifier'], ['oblique', 'temporal', 'modifier'], ['oblique', 'nominal'], ['orphan'], ['parataxis'], ['punct'], ['overridden', 'disfluency'], ['root'], ['vocative'], ['open', 'clausal', 'complement']
]

target_tokens = ['[ADJ]', '[ADP]', '[ADV]', '[AUX]', '[CCONJ]', '[DET]', '[INTJ]', '[NOUN]', '[NUM]', '[PART]', '[PRON]', '[PROPN]', '[PUNCT]', '[SCONJ]', '[SYM]', '[VERB]', '[X]', 
    '[acl:cleft]', '[acl:relcl]', '[acl]', '[advcl:cleft]', '[advcl:tcl]', '[advcl]', '[advmod:emph]', '[advmod:tmod]', '[advmod]', '[amod]', '[appos]', '[aux:caus]', '[aux:pass]', '[aux]', '[case]', '[cc:preconj]', '[cc]', '[ccomp:pmod]', '[ccomp]', '[compound:prt]', '[compound]', '[conj]', '[cop]', '[csubj:pass]', '[csubj]', '[dep]', '[det:numgov]', '[det:nummod]', '[det:poss]', '[det:predet]', '[det]', '[discourse]', '[dislocated]', '[expl:impers]', '[expl:pass]', '[expl:poss]', '[expl:pv]', '[expl]', '[fixed]', '[flat:foreign]', '[flat:name]', '[flat]', '[goeswith]', '[iobj:agent]', '[iobj]', '[list]', '[mark]', '[nmod:agent]', '[nmod:npmod]', '[nmod:pmod]', '[nmod:poss]', '[nmod:range]', '[nmod:tmod]', '[nmod]', '[nsubj:caus]', '[nsubj:pass]', '[nsubj]', '[nummod:entity]', '[nummod:gov]', '[nummod]', '[obj:agent]', '[obj]', '[obl:agent]', '[obl:arg]', '[obl:mod]', '[obl:npmod]', '[obl:tmod]', '[obl]', '[orphan]', '[parataxis]', '[punct]', '[reparandum]', '[root]', '[vocative]', '[xcomp]'
]



def clean(seq):
    seq = seq.strip()
    seq = re.sub('<pad>', '', seq)
    seq = re.sub('</s>', '', seq)
    seq = re.sub('\n', '', seq)
    seq = re.sub(' ', '', seq)
    return seq

def metric(tokenizer, predictions, golds, best_metric, extra_labels=None):
    dev_arc, dev_rel, dev_total_pred, dev_total_gold = 0, 0, 0, 0
    test_arc, test_rel, test_total_pred, test_total_gold = 0, 0, 0, 0

    with open('ud-bg-2000.txt', 'w') as fw:
        for sen_idx, (pred, gold) in enumerate(zip(predictions, golds)):
            pred, gold = clean(pred), clean(gold)
            fw.write(pred + '\n')
            fw.write(gold + '\n')
            pred_len = pred.count(split_token)
            pred_chunks = pred.split(split_token)
            gold_len = gold.count(split_token)
            gold_chunks = gold.split(split_token)

            if sen_idx < 1115:
                dev_total_gold += gold_len
                dev_total_pred += pred_len

                for idx in range(min(pred_len, gold_len)):
                    pred_chunk, gold_chunk = pred_chunks[idx], gold_chunks[idx]
                    if pred_chunk == gold_chunk:
                        dev_rel += 1
                    if ']' in pred_chunk:
                        pred_head = pred_chunk.split(']')[1]
                        gold_head = gold_chunk.split(']')[1]
                        if pred_head == gold_head:
                            dev_arc += 1

            if sen_idx >= 1115:
                test_total_gold += gold_len
                test_total_pred += pred_len

                for idx in range(min(pred_len, gold_len)):
                    pred_chunk, gold_chunk = pred_chunks[idx], gold_chunks[idx]
                    if pred_chunk == gold_chunk:
                        test_rel += 1
                    if ']' in pred_chunk:
                        pred_head = pred_chunk.split(']')[1]
                        gold_head = gold_chunk.split(']')[1]
                        if pred_head == gold_head:
                            test_arc += 1
    
    if dev_total_pred == 0 or dev_total_gold == 0:
        dev_uas, dev_las = 0, 0
    else:
        dev_uas = dev_arc / dev_total_gold * 100
        dev_las = dev_rel / dev_total_gold * 100

    if test_total_pred == 0 or test_total_gold == 0:
        test_uas, test_las = 0, 0
    else:
        test_uas = test_arc / test_total_gold * 100
        test_las = test_rel / test_total_gold * 100
        
    if best_metric:
        if dev_las > best_metric:
            print("current best dev_las=%.2f, test_las=%.2f" % (dev_las, test_las))
    else:
        print("current best dev_las=%.2f, test_las=%.2f" % (dev_las, test_las))
    print("dev_uas=%d/%d=%.2f dev_las=%d/%d=%.2f" % (dev_arc, dev_total_gold, dev_uas, dev_rel, dev_total_gold, dev_las))
    print("test_uas=%d/%d=%.2f test_las=%d/%d=%.2f" % (test_arc, test_total_gold, test_uas, test_rel, test_total_gold, test_las))
    return {
        "dev_uas": dev_uas,
        "dev_las": dev_las,
        "test_uas": test_uas,
        "test_las": test_las
    }
